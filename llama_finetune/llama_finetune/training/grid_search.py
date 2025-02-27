import os
from datetime import datetime
import json
import numpy as np
from datasets import load_dataset
from .model_loader import load_model_and_tokenizer
from .trainer_setup import setup_trainer
from .config import get_peft_params, get_training_params
from .training_utils import process_trained_model
from .smac_optimizer import optimize_hyperparameters, split_config_dict
from .plotting_utils import plot_comparison

def execute_smac_optimization(args, model, tokenizer, max_seq_length, train_dataset_size, n_trials=20):
    """Execute SMAC optimization for hyperparameter tuning."""
    
    def objective_function(config, seed: int = 42):  # Default seed value
        """Objective function for SMAC optimization"""
        try:
            # Load fresh model for each trial
            current_model, current_tokenizer = load_model_and_tokenizer(
                model_name="unsloth/Llama-3.2-3B-Instruct",
                max_seq_length=max_seq_length
            )
            
            # Split config into PEFT and training parameters
            peft_updates, training_updates = split_config_dict(config)
            
            # Add seed to training parameters
            training_updates["seed"] = seed
            
            # Get full parameter dictionaries
            current_peft_params = get_peft_params(peft_updates)
            current_training_params = get_training_params(training_updates)
            
            # Create output directory for this trial
            trial_output_dir = os.path.join(
                args.output_dir,
                f"smac_trial_r{peft_updates['r']}_alpha{peft_updates['lora_alpha']}"
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(trial_output_dir, exist_ok=True)
            
            # Run training
            process_trained_model(
                args, max_seq_length, current_model, trial_output_dir,
                current_tokenizer, train_dataset_size,
                current_peft_params, current_training_params
            )
            
            # Load and process evaluation results
            eval_results_path = os.path.join(trial_output_dir, "evaluation_results.json")
            with open(eval_results_path, "r") as f:
                eval_results = json.load(f)
            
            execution_metrics = eval_results.get("execution_check", {})
            running_snippets = execution_metrics.get("successful_runs", 0)
            total_snippets = execution_metrics.get("total_snippets", 1)
            execution_rate = running_snippets / total_snippets
            avg_bleu = eval_results.get("avg_bleu", 0)
            
            # Combine metrics into single score (higher is better)
            score = (0.7 * execution_rate) + (0.3 * avg_bleu)
            
            # SMAC minimizes, so return negative score
            return -score
            
        except Exception as e:
            print(f"Error in trial: {str(e)}")
            raise e
    
    # Run SMAC optimization
    incumbent, runhistory = optimize_hyperparameters(objective_function, n_trials=n_trials)
    
    # Process results
    best_config = incumbent.get_dictionary()
    peft_updates, training_updates = split_config_dict(best_config)
    
    # Get the best score from runhistory
    best_score = -runhistory.get_cost(incumbent)  # Negate back since we minimized
    running_rate = best_score * 0.7  # Extract execution rate component
    bleu_score = best_score * 0.3   # Extract BLEU score component
    
    # Generate comparison plot of all trials
    # Get all trial directories
    trial_dirs = [
        d for d in os.listdir(args.output_dir)
        if d.startswith("smac_trial_") and os.path.isdir(os.path.join(args.output_dir, d))
    ]
    
    if trial_dirs:
        comparison_plot = plot_comparison(
            trial_dirs,
            args.output_dir,
            os.path.join(args.output_dir, "comparisons")
        )
        if comparison_plot:
            print(f"Comparison plot saved to: {comparison_plot}")
    
    return {
        "peft": peft_updates,
        "training": training_updates,
        "execution_rate": running_rate,
        "bleu": bleu_score,
        "running_snippets": int(running_rate * 17),  # Assuming 17 test cases
        "total_snippets": 17,
        "timestamp": datetime.now().isoformat()
    }

