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



from config import PEFT_PARAMS, TRAINING_PARAMS, store_model_info



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


def parse_args():
    import argparse

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
    return parser.parse_args()

def process_loaded_model(args, model, output_dir, tokenizer, train_dataset_size) -> None:
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


def process_trained_model(args, max_seq_length, model, output_dir, tokenizer, train_dataset_size):
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


# For fine tuning:
#   poetry run python src/train.py
# For loading fine-tuned model:
#   poetry run python src/train.py --load-model ./res/outputs/finetuned_model
def main():

    #  TODO: ADD ASSERT ABOUT THE CURRENT WORK DIRECTORY: THE SCRIPT SHOULD BE EXECUTED INSIDE llama_finetune/llama_finetune
    # Parse command line arguments

    args = parse_args()

    print("Args parsed!")

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
        model, output_prefix, results_file = process_loaded_model(args, model, output_dir, tokenizer, train_dataset_size)
    else:
        process_trained_model(args, max_seq_length, model, output_dir, tokenizer, train_dataset_size)

if __name__ == "__main__":
    main()
