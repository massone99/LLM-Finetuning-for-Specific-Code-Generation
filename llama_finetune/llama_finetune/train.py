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

from evaluate import evaluate_model, extract_generated_code
from logger import file_logger
from termcolor import colored

import hashlib
import json
from datetime import datetime

from training.config import PEFT_PARAMS, TRAINING_PARAMS, store_model_info
from training.config import (
    get_grid_combinations,
    get_peft_params,
    get_training_params,
)

from training.model_loader import load_model_and_tokenizer
from training.dataset_utils import prepare_dataset
from training.trainer_setup import setup_trainer
from training.grid_search import execute_grid_search


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
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable grid search for hyperparameter tuning",
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


def process_trained_model(args, max_seq_length, model, output_dir, tokenizer, train_dataset_size, peft_params=None, training_params=None):
    # Prepare dataset
    dataset = prepare_dataset(args.dataset_path, tokenizer)

    # Setup and start training with specific parameters
    trainer = setup_trainer(
        model, 
        tokenizer, 
        dataset, 
        max_seq_length, 
        output_dir,
        peft_params,
        training_params
    )

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


def save_grid_search_results(output_dir: str, best_params: dict) -> None:
    """Save grid search results to a JSON file."""
    if not best_params:
        return

    print("\nBest performing combination:")
    print(f"Execution rate: {best_params['execution_rate']:.2%} "
          f"({best_params['running_snippets']}/{best_params['total_snippets']})")
    print(f"BLEU score: {best_params['bleu']:.4f}")
    print(f"PEFT parameters: {best_params['peft']}")
    print(f"Training parameters: {best_params['training']}")

    results_dir = os.path.join(output_dir, "grid_search_results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(
        results_dir, 
        f'best_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(results_file, "w") as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\nBest parameters saved to: {results_file}")

# Training normale
# poetry run python train.py --dataset-path ../res/data/dataset_llama.json --output-dir ../res/outputs

# Grid search
# poetry run python train.py --dataset-path ../res/data/dataset_llama.json --output-dir ../res/outputs --grid-search

# Caricamento di un modello esistente
# poetry run python train.py --load-model ../res/outputs/finetuned_model
def main():
    # Verify working directory
    current_dir = os.path.basename(os.getcwd())
    parent_dir = os.path.basename(os.path.dirname(os.getcwd()))
    if not (current_dir == "llama_finetune" and parent_dir == "llama_finetune"):
        raise RuntimeError("This script must be run from inside the llama_finetune/llama_finetune directory")
    
    print("Current working directory: CORRECT")
    
    args = parse_args()
    print("Args parsed!")

    # Configuration
    max_seq_length = 4096
    output_dir = args.output_dir

    # Load initial model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length
    )

    # Get training dataset size
    try:
        train_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
        train_dataset_size = len(train_dataset)
    except Exception as e:
        print(f"Warning: Could not determine training dataset size: {e}")
        train_dataset_size = None

    try:
        if args.load_model:
            process_loaded_model(args, model, output_dir, tokenizer, train_dataset_size)
        elif args.grid_search:
            best_params = execute_grid_search(args, model, tokenizer, max_seq_length, train_dataset_size)
            save_grid_search_results(output_dir, best_params)
        else:
            # Single training run with default parameters
            process_trained_model(
                args,
                max_seq_length,
                model,
                output_dir,
                tokenizer,
                train_dataset_size,
                PEFT_PARAMS,
                TRAINING_PARAMS
            )
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
