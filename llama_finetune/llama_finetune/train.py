from datasets import load_dataset
from peft import PeftModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only

from evaluate import extract_generated_code
from logger import file_logger


def load_model_and_tokenizer(model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=2048):
    """Load the model and tokenizer using the llama chat template"""
    dtype = None  # Auto detection
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=max_seq_length,
        dtype=dtype, load_in_4bit=load_in_4bit, )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def prepare_dataset(dataset_path, tokenizer):
    """Prepare the dataset for training."""
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir="outputs"):
    """Set up the SFT trainer."""
    # Here we are using LORA with PEFT
    model = FastLanguageModel.get_peft_model(model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_alpha=16,
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407, use_rslora=False,
        loftq_config=None, )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text",
        max_seq_length=max_seq_length, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer), dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=5,
            max_steps=60, learning_rate=2e-4, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            logging_steps=1, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear", seed=3407,
            output_dir=output_dir, report_to="none", ), )

    trainer = train_on_responses_only(trainer, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n", )

    return trainer


# For fine tuning:
#   poetry run python src/train.py
# For loading fine-tuned model:
#   poetry run python src/train.py --load-model ./res/outputs/finetuned_model
def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or load a fine-tuned model')
    parser.add_argument('--load-model', type=str, help='Path to load fine-tuned model from', default=None)
    parser.add_argument('--dataset-path', type=str, default="./res/data/dataset_llama.json",
                        help='Path to training dataset')
    parser.add_argument('--test-dataset-path', type=str, default="./res/data/test_set.json",
                        help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default="./res/outputs", help='Directory for output files')
    args = parser.parse_args()

    # Configuration
    max_seq_length = 4096
    output_dir = args.output_dir

    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(max_seq_length=max_seq_length)

    # Get training dataset size from the original training data
    try:
        train_dataset = load_dataset('json', data_files=args.dataset_path, split='train')
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
        evaluate_model(model, tokenizer, args.test_dataset_path, train_dataset_size, output_prefix)
    else:
        # Evaluate model before fine-tuning
        print("Evaluating model before fine-tuning...")

        from evaluate import evaluate_model
        output_prefix = "before_finetuning"
        evaluate_model(model, tokenizer, args.test_dataset_path, train_dataset_size, output_prefix)

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
        training_metrics = {"training_time_seconds": training_time,
            "training_stats": training_output.metrics if training_output else None}
        import json
        with open(f"{output_dir}/training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2)

        # Save the fine-tuned model
        trainer.save_model(f"{output_dir}/finetuned_model")

        # Evaluate model after fine-tuning
        print("Evaluating model after fine-tuning...")
        output_prefix = f"after_finetuning_trainsize{train_dataset_size}"
        evaluate_model(model, tokenizer, args.test_dataset_path, train_dataset_size, output_prefix)
        train_time_str = f"\nTraining completed in {training_time:.2f} seconds"
        train_metrics_path_str = f"Training metrics saved to: {output_dir}/training_metrics.json"

        file_logger.write_and_print(train_time_str)
        file_logger.write_and_print(train_metrics_path_str)


if __name__ == "__main__":
    main()
