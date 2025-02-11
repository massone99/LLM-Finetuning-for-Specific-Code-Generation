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



from config import PEFT_PARAMS, TRAINING_PARAMS, store_model_info
from config import (
    get_grid_combinations,
    get_peft_params,
    get_training_params,
)


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


def setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir="outputs", peft_params=None, training_params=None):
    """Set up the SFT trainer with specific parameters for grid search"""
    # Use provided parameters or fallback to globals
    peft_params = peft_params or PEFT_PARAMS
    training_params = training_params or TRAINING_PARAMS

    # Use the provided PEFT parameters
    model = FastLanguageModel.get_peft_model(model, **peft_params)

    # Update output_dir in training params
    local_training_params = training_params.copy()
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


# For fine tuning:
#   poetry run python src/train.py
# For loading fine-tuned model:
#   poetry run python src/train.py --load-model ./res/outputs/finetuned_model
def main():

    current_dir = os.path.basename(os.getcwd())
    assert current_dir == "llama_finetune", "This script must be run from inside the llama_finetune/llama_finetune directory"
    parent_dir = os.path.basename(os.path.dirname(os.getcwd()))
    assert parent_dir == "llama_finetune", "This script must be run from inside the llama_finetune/llama_finetune directory"
    
    print("Current working directory: CORRECT")
    
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
        process_loaded_model(args, model, output_dir, tokenizer, train_dataset_size)
    else:
        # Evaluate base model once before starting grid search
        # print("Evaluating base model before starting grid search...")
        # from evaluate import evaluate_model
        # evaluate_model(
        #     model, tokenizer, args.test_dataset_path, train_dataset_size, "before_finetuning"
        # )

        # Grid search implementation
        combinations = get_grid_combinations()
        print(f"Starting grid search with {len(combinations)} combinations...")

        best_execution_rate = 0
        best_params = None

        for i, (peft_updates, training_updates) in enumerate(combinations):
            print(f"\nRunning combination {i+1}/{len(combinations)}")
            print(f"PEFT params: {peft_updates}")
            print(f"Training params: {training_updates}")

            # Get complete parameter sets
            current_peft_params = get_peft_params(peft_updates)
            current_training_params = get_training_params(training_updates)

            # Update output directory for this combination
            combo_output_dir = os.path.join(
                args.output_dir,
                f"combo_{i+1}_r{peft_updates['r']}_alpha{peft_updates['lora_alpha']}_"
                f"bs{training_updates['per_device_train_batch_size']}_"
                f"ep{training_updates['num_train_epochs']}_"
                f"lr{training_updates['learning_rate']}"
            )

            # Load fresh model for each combination
            model, tokenizer = load_model_and_tokenizer(
                model_name="unsloth/Llama-3.2-3B-Instruct",
                max_seq_length=max_seq_length
            )

            # Train and evaluate
            try:
                process_trained_model(
                    args,
                    max_seq_length,
                    model,
                    combo_output_dir,
                    tokenizer,
                    train_dataset_size,
                    current_peft_params,    # Pass the current parameters
                    current_training_params,  # Pass the current parameters
                )

                # Load the evaluation results
                with open(os.path.join(combo_output_dir, "evaluation_results.json"), "r") as f:
                    eval_results = json.load(f)
                
                # Calculate execution rate
                running_snippets = eval_results.get("execution_check", {}).get("successful_runs", 0)
                total_snippets = eval_results.get("execution_check", {}).get("total_snippets", 1)  # avoid div by 0
                execution_rate = running_snippets / total_snippets if total_snippets > 0 else 0
                
                print(f"Execution rate: {execution_rate:.2%} ({running_snippets}/{total_snippets})")
                print(f"BLEU score: {eval_results.get('avg_bleu', 0):.4f}")
                
                if execution_rate > best_execution_rate:
                    best_execution_rate = execution_rate
                    best_params = {
                        "peft": peft_updates,
                        "training": training_updates,
                        "execution_rate": execution_rate,
                        "running_snippets": running_snippets,
                        "total_snippets": total_snippets,
                        "bleu": eval_results.get("avg_bleu", 0)
                    }
            
            except Exception as e:
                print(f"Error in combination {i+1}: {str(e)}")
                continue

        # Print best results and save to JSON
        if best_params:
            print("\nBest performing combination:")
            print(f"Execution rate: {best_params['execution_rate']:.2%} "
                  f"({best_params['running_snippets']}/{best_params['total_snippets']})")
            print(f"BLEU score: {best_params['bleu']:.4f}")
            print(f"PEFT parameters: {best_params['peft']}")
            print(f"Training parameters: {best_params['training']}")

            # Add timestamp to best_params
            best_params["timestamp"] = datetime.now().isoformat()
            
            # Create results directory if it doesn't exist
            results_dir = os.path.join(output_dir, "grid_search_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save best params to JSON file
            results_file = os.path.join(
                results_dir, 
                f'best_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(results_file, "w") as f:
                json.dump(best_params, f, indent=2)
            
            print(f"\nBest parameters saved to: {results_file}")
if __name__ == "__main__":
    main()
