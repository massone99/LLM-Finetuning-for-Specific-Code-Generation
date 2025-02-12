import os
from datetime import datetime
import json
from datasets import load_dataset
from .model_loader import load_model_and_tokenizer
from .trainer_setup import setup_trainer
from .config import get_grid_combinations, get_peft_params, get_training_params
from .training_utils import process_trained_model

def execute_grid_search(args, model, tokenizer, max_seq_length, train_dataset_size):
    """Execute grid search training with different parameter combinations."""
    combinations = get_grid_combinations()
    print(f"Starting grid search with {len(combinations)} combinations...")

    best_execution_rate = -1  # Changed from 0 to -1 to ensure first valid result is captured
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

            # Run training for this combination
            process_trained_model(
                args, max_seq_length, current_model, combo_output_dir,
                current_tokenizer, train_dataset_size,
                current_peft_params, current_training_params
            )

            # Process evaluation results with validation
            eval_results_path = os.path.join(combo_output_dir, "evaluation_results.json")
            if not os.path.exists(eval_results_path):
                print(f"Warning: No evaluation results found at {eval_results_path}")
                continue
                
            try:
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)
                
                execution_metrics = eval_results.get("execution_check", {})
                running_snippets = execution_metrics.get("successful_runs")
                total_snippets = execution_metrics.get("total_snippets")
                
                # Validate all metrics before processing
                if running_snippets is None or total_snippets is None:
                    print(f"Warning: Missing execution metrics in {eval_results_path}")
                    continue
                
                # Validate metrics types and values
                if not isinstance(running_snippets, (int, float)) or not isinstance(total_snippets, (int, float)):
                    print(f"Warning: Invalid metrics format in {eval_results_path}")
                    continue
                    
                if total_snippets <= 0:
                    print(f"Warning: Invalid total_snippets value: {total_snippets}")
                    continue
                
                execution_rate = running_snippets / total_snippets
                avg_bleu = eval_results.get("avg_bleu")
                
                if avg_bleu is None:
                    print(f"Warning: Missing BLEU score in {eval_results_path}")
                    avg_bleu = 0
                elif not isinstance(avg_bleu, (int, float)):
                    best_params = {
                        "peft": peft_updates,
                        "training": training_updates,
                        "execution_rate": execution_rate,
                        "running_snippets": running_snippets,
                        "total_snippets": total_snippets,
                        "bleu": avg_bleu,
                        "timestamp": datetime.now().isoformat()
                    }
                
            except json.JSONDecodeError as e:
                print(f"Error reading evaluation results: {str(e)}")
                continue
                
        except Exception as e:
            raise e
            print(f"Error in combination {i+1}: {str(e)}")
            continue

    if best_params is None:
        print("\nWarning: No valid results found during grid search")
    
    return best_params
