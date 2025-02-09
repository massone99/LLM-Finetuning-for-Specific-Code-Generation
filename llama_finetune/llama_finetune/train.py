import os
from datasets import load_dataset
from peft import PeftModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    train_on_responses_only,
)

from evaluate import extract_generated_code
from logger import file_logger
from termcolor import colored

import hashlib
import json
from datetime import datetime


# PEFT (Parameter-Efficient Fine-Tuning) parameters - results gathered from experiments
PEFT_PARAMS = {
    "r": 64,  # LoRA rank - 64: Running examples 6/17 - 128: crashes PC
    "target_modules": [  # Layers to apply LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # Attention layers
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP layers
        # "lm_head"            # Optional: language model head
    ],
    # lora_alpha results:
    # - 16: basic performance
    # - 128: good performance
    # - 512: best performance, BLEU: 0.36-0.37, Running examples: 8-10/17
    # - 1024: worse performance, Running examples: 7/17
    "lora_alpha": 512,  # α > rank amplifies updates influence
    "lora_dropout": 0,  # Optimized when 0
    "bias": "none",  # Optimized when "none"
    "use_gradient_checkpointing": "unsloth",  # 30% less VRAM usage
    "random_state": 3407,  # Fixed seed for reproducibility
    "use_rslora": False,  # Rank-stabilized LoRA disabled
    "loftq_config": None,  # LoftQ quantization disabled
}

# Training hyperparameters - results gathered from experiments
TRAINING_PARAMS = {
    # Batch sizes tested:
    # - 2: works but slow
    # - 4: good balance
    # - 8: crashes due to memory
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    # Steps configuration:
    # - max_steps=500, warmup_steps=50: mediocre results (BLEU: 0.38, 5/17)
    # - max_steps=60, warmup_steps=5: better results
    "warmup_steps": 5,
    "num_train_epochs": 5,
    "max_steps": 60,
    "learning_rate": 2e-4,
    "fp16": not is_bfloat16_supported(),  # Use FP16 if bfloat16 not available
    "bf16": is_bfloat16_supported(),  # Prefer bfloat16 if supported
    "logging_steps": 1,
    "optim": "adamw_8bit",  # 8-bit AdamW for memory efficiency
    "weight_decay": 0.01,  # L2 regularization
    "lr_scheduler_type": "linear",  # Linear learning rate decay
    "seed": 3407,  # Fixed seed for reproducibility
    "report_to": "none",  # Disable external logging
}


def store_model_info(output_dir, train_dataset_size, test_dataset_size, trainer_args, avg_bleu, samples_info, peft_params=PEFT_PARAMS):
    """
    Compute a unique hash for the model and hyperparameters, and append
    model info to 'trained_models.json' if this hash isn't present yet.
    """
    # Combine relevant info into a single string, including PEFT params
    info_str = (
        f"batch_size={trainer_args.per_device_train_batch_size}-"
        f"lr={trainer_args.learning_rate}-"
        f"epochs={trainer_args.num_train_epochs}-"
        f"max_steps={trainer_args.max_steps}-"
        f"train_size={train_dataset_size}-"
        f"test_size={test_dataset_size}-"
        f"lora_rank={peft_params['r']}-"
        f"lora_alpha={peft_params['lora_alpha']}-"
        f"lora_dropout={peft_params['lora_dropout']}"
    )
    model_hash = hashlib.sha256(info_str.encode("utf-8")).hexdigest()

    record = {
        "model_hash": model_hash,
        "timestamp": datetime.now().isoformat(),
        # Training parameters
        "batch_size": trainer_args.per_device_train_batch_size,
        "gradient_accumulation_steps": trainer_args.gradient_accumulation_steps,
        "learning_rate": trainer_args.learning_rate,
        "num_train_epochs": trainer_args.num_train_epochs,
        "max_steps": trainer_args.max_steps,
        "warmup_steps": trainer_args.warmup_steps,
        "optimizer": trainer_args.optim,
        "weight_decay": trainer_args.weight_decay,
        "lr_scheduler": trainer_args.lr_scheduler_type,
        # PEFT parameters
        "lora_rank": peft_params["r"],
        "lora_alpha": peft_params["lora_alpha"],
        "lora_dropout": peft_params["lora_dropout"],
        "target_modules": peft_params["target_modules"],
        "use_gradient_checkpointing": peft_params["use_gradient_checkpointing"],
        # Dataset info
        "train_dataset_size": train_dataset_size,
        "test_dataset_size": test_dataset_size,
        # Evaluation metrics
        "avg_bleu": avg_bleu,
        "execution_check": {
            "successful_runs": samples_info[0],
            "total_snippets": samples_info[1]
        },  
    }

    trained_models_path = os.path.join(output_dir, "trained_models.json")

    # Load existing records, or create a new list
    if os.path.exists(trained_models_path):
        with open(trained_models_path, "r") as f:
            trained_models = json.load(f)
    else:
        trained_models = []
        os.makedirs(os.path.dirname(trained_models_path), exist_ok=True)

    # Check if hash already present
    if any(m["model_hash"] == model_hash for m in trained_models):
        print(f"Model info already exists for hash: {model_hash}")
    else:
        trained_models.append(record)
        with open(trained_models_path, "w") as f:
            json.dump(trained_models, f, indent=2)
        print(f"Stored new model info with hash: {model_hash}")


def load_model_and_tokenizer(
    model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=2048
):
    """Load the model and tokenizer using the llama chat template"""

    print(colored(f"Finetuning model: {model_name}", "green"))

    dtype = None  # Auto detection
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def prepare_dataset(dataset_path, tokenizer):
    """Prepare the dataset for training."""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir="outputs"):
    """Set up the SFT trainer."""
    # Use global PEFT parameters
    model = FastLanguageModel.get_peft_model(model, **PEFT_PARAMS)

    # Update output_dir in training params
    local_training_params = TRAINING_PARAMS.copy()
    local_training_params["output_dir"] = output_dir

    # Create training arguments
    training_args = TrainingArguments(**local_training_params)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    return trainer


# For fine tuning:
#   poetry run python src/train.py
# For loading fine-tuned model:
#   poetry run python src/train.py --load-model ./res/outputs/finetuned_model
def main():
    import argparse

    #  TODO: ADD ASSERT ABOUT THE CURRENT WORK DIRECTORY: THE SCRIPT SHOULD BE EXECUTED INSIDE llama_finetune/llama_finetune
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train or load a fine-tuned model")
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to load fine-tuned model from",
        default=None,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../res/data/dataset_llama.json",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=str,
        default="../res/data/test_set.json",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../res/outputs",
        help="Directory for output files",
    )
    args = parser.parse_args()

    # Configuration
    max_seq_length = 4096
    output_dir = args.output_dir

    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=max_seq_length
    )

    # Get training dataset size from the original training data
    try:
        train_dataset = load_dataset(
            "json", data_files=args.dataset_path, split="train"
        )
        train_dataset_size = len(train_dataset)
    except Exception as e:
        print(f"Warning: Could not determine training dataset size: {e}")
        train_dataset_size = None

    if args.load_model:
        print(f"Loading fine-tuned model from {args.load_model}...")
        # Load the PEFT model
        model = PeftModel.from_pretrained(model, args.load_model)

        if train_dataset_size:
            output_prefix = f"loaded_finetuned_trainsize{train_dataset_size}"
        else:
            output_prefix = "loaded_finetuned"

        # Evaluate loaded model
        print("Evaluating loaded fine-tuned model...")
        from evaluate import evaluate_model

        results_file, avg_bleu, samples_info = evaluate_model(
            model, tokenizer, args.test_dataset_path, train_dataset_size, output_prefix
        )

        # Create training arguments with global parameters
        local_training_params = TRAINING_PARAMS.copy()
        local_training_params["output_dir"] = output_dir
        training_args = TrainingArguments(**local_training_params)

        store_model_info(
            "../res/data/trained_models/",
            train_dataset_size,
            len(load_dataset("json", data_files=args.test_dataset_path, split="train")),
            training_args,
            avg_bleu,
            samples_info,
            PEFT_PARAMS
        )

    else:
        # Evaluate model before fine-tuning
        print("Evaluating model before fine-tuning...")

        from evaluate import evaluate_model

        output_prefix = "before_finetuning"
        
        evaluate_model(
            model, tokenizer, args.test_dataset_path, train_dataset_size, output_prefix
        )

        # Prepare dataset
        dataset = prepare_dataset(args.dataset_path, tokenizer)

        # Setup and start training
        trainer = setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir)

        # Track training time
        import time

        start_time = time.time()
        training_output = trainer.train()
        training_time = time.time() - start_time

        # Save training metrics
        training_metrics = {
            "training_time_seconds": training_time,
            "training_stats": training_output.metrics if training_output else None,
        }
        import json

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2)

        # Save the fine-tuned model
        trainer.save_model(f"{output_dir}/finetuned_model")

        # Evaluate model after fine-tuning
        print("Evaluating model after fine-tuning...")
        output_prefix = f"after_finetuning_trainsize{train_dataset_size}"
        
        # FIXME
        results_file, avg_bleu, samples_info = evaluate_model(
            model, tokenizer, args.test_dataset_path, train_dataset_size, output_prefix
        )
        
        train_time_str = f"\nTraining completed in {training_time:.2f} seconds"
        train_metrics_path_str = (
            f"\nTraining metrics saved to: {output_dir}/training_metrics.json"
        )

        # Store model info
        store_model_info(
            "../res/data/trained_models/",
            train_dataset_size,
            len(load_dataset("json", data_files=args.test_dataset_path, split="train")),
            trainer.args,
            avg_bleu,
            samples_info,
            PEFT_PARAMS
        )

        file_logger.write_and_print(train_time_str)
        file_logger.write_and_print(train_metrics_path_str)


if __name__ == "__main__":
    main()
