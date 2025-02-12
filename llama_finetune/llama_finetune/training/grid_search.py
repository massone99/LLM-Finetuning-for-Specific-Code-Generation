import os
from datetime import datetime
import json
from datasets import load_dataset
from .model_loader import load_model_and_tokenizer
from .trainer_setup import setup_trainer
from .config import get_grid_combinations, get_peft_params, get_training_params

def execute_grid_search(args, model, tokenizer, max_seq_length, train_dataset_size):
    """Execute grid search training with different parameter combinations."""
    combinations = get_grid_combinations()
    print(f"Starting grid search with {len(combinations)} combinations...")

    best_execution_rate = 0
    best_params = None

    for i, (peft_updates, training_updates) in enumerate(combinations):
        print(f"\nRunning combination {i+1}/{len(combinations)}")
        print(f"PEFT params: {peft_updates}")
        print(f"Training params: {training_updates}")

        combo_output_dir = os.path.join(
            args.output_dir,
            f"combo_{i+1}_r{peft_updates['r']}_alpha{peft_updates['lora_alpha']}_"
            f"bs{training_updates['per_device_train_batch_size']}_"
            f"ep{training_updates['num_train_epochs']}_"
            f"lr{training_updates['learning_rate']}"
        )

        try:
            # Load fresh model for each combination
            current_model, current_tokenizer = load_model_and_tokenizer(
                model_name="unsloth/Llama-3.2-3B-Instruct",
                max_seq_length=max_seq_length
            )

            current_peft_params = get_peft_params(peft_updates)
            current_training_params = get_training_params(training_updates)

            process_trained_model(
                args, max_seq_length, current_model, combo_output_dir,
                current_tokenizer, train_dataset_size,
                current_peft_params, current_training_params
            )

            # Process evaluation results
            eval_results_path = os.path.join(combo_output_dir, "evaluation_results.json")
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
            
            execution_metrics = eval_results.get("execution_check", {})
            running_snippets = execution_metrics.get("successful_runs", 0)
            total_snippets = execution_metrics.get("total_snippets", 1)
            execution_rate = running_snippets / total_snippets if total_snippets > 0 else 0
            
            print(f"Execution rate: {execution_rate:.2%} ({running_snippets}/{total_snippets})")
            print(f"BLEU score: {eval_results.get('avg_bleu', 0)::.4f}")
            
            if execution_rate > best_execution_rate:
                best_execution_rate = execution_rate
                best_params = {
                    "peft": peft_updates,
                    "training": training_updates,
                    "execution_rate": execution_rate,
                    "running_snippets": running_snippets,
                    "total_snippets": total_snippets,
                    "bleu": eval_results.get("avg_bleu", 0),
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            print(f"Error in combination {i+1}: {str(e)}")
            continue

    return best_params
