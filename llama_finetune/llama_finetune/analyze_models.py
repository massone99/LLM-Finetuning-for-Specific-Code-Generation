import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import argparse
from termcolor import colored
import matplotlib.pyplot as plt

def load_trained_models(file_path: str) -> List[Dict[str, Any]]:
    """Load trained models data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_available_dataset_sizes(models: List[Dict[str, Any]]) -> List[int]:
    """Get list of unique dataset sizes used in training."""
    return sorted(list(set(model['train_dataset_size'] for model in models)))

def calculate_success_rate(model: Dict[str, Any]) -> float:
    """Calculate success rate from execution check."""
    exec_check = model['execution_check']
    return exec_check['successful_runs'] / exec_check['total_snippets']

def find_best_model_for_size(models: List[Dict[str, Any]], dataset_size: int) -> Dict[str, Any]:
    """Find the best performing model for a given dataset size, prioritizing success rate."""
    size_models = [m for m in models if m['train_dataset_size'] == dataset_size]
    
    if not size_models:
        return None
    
    # Sort primarily by success rate, use BLEU score only as a tiebreaker
    sorted_models = sorted(
        size_models,
        key=lambda x: (calculate_success_rate(x), x['avg_bleu']),
        reverse=True
    )
    
    return sorted_models[0]

def get_size_statistics(models: List[Dict[str, Any]], dataset_size: int) -> Dict[str, Any]:
    """Calculate success rate statistics for a specific dataset size."""
    size_models = [m for m in models if m['train_dataset_size'] == dataset_size]
    success_rates = [calculate_success_rate(m) for m in size_models]
    
    if not success_rates:
        return None
        
    return {
        "max_rate": max(success_rates),
        "avg_rate": sum(success_rates) / len(success_rates),
        "min_rate": min(success_rates),
        "num_models": len(success_rates)
    }

def print_model_params(model: Dict[str, Any]):
    """Print relevant model parameters in a formatted way, emphasizing success rate."""
    success_rate = calculate_success_rate(model)
    
    print(colored("\nModel Performance:", "green"))
    print(colored(f"Success Rate: {success_rate:.2%} ({model['execution_check']['successful_runs']}/{model['execution_check']['total_snippets']})", "yellow", attrs=['bold']))
    print(f"BLEU Score: {model['avg_bleu']:.4f}")
    print(f"Training Dataset Size: {model['train_dataset_size']}")
    
    print("\nTraining Parameters:")
    print(f"  Learning Rate: {model['learning_rate']}")
    print(f"  Batch Size: {model['batch_size']}")
    print(f"  Epochs: {model['num_train_epochs']}")
    print(f"  Gradient Accumulation Steps: {model['gradient_accumulation_steps']}")
    
    print("\nLoRA Parameters:")
    print(f"  Rank: {model['lora_rank']}")
    print(f"  Alpha: {model['lora_alpha']}")
    print(f"  Dropout: {model['lora_dropout']}")
    print(f"\nTimestamp: {model['timestamp']}")

def count_models_by_size(models: List[Dict[str, Any]]) -> Dict[int, int]:
    """Count how many models exist for each dataset size."""
    counts = defaultdict(int)
    for model in models:
        counts[model['train_dataset_size']] += 1
    return counts

def analyze_performance_trend(models: List[Dict[str, Any]]) -> None:
    """Analyze how performance changes with dataset size."""
    # Sort models by dataset size and get best performance for each size
    size_to_best_rate = {}
    for model in models:
        size = model['train_dataset_size']
        rate = calculate_success_rate(model)
        if size not in size_to_best_rate or rate > size_to_best_rate[size]:
            size_to_best_rate[size] = rate
    
    # Find best performing size
    best_size = max(size_to_best_rate.items(), key=lambda x: x[1])[0]
    
    print(colored("\n=== Performance Analysis ===", "yellow"))
    print(f"Best performance achieved with dataset size: {best_size}\n")
    print("Performance changes:")
    
    # Print performance changes between consecutive sizes
    sizes = sorted(size_to_best_rate.keys())
    for i in range(1, len(sizes)):
        prev_size = sizes[i-1]
        curr_size = sizes[i]
        delta = size_to_best_rate[curr_size] - size_to_best_rate[prev_size]
        delta_str = f"{delta:+.2%}"
        color = "green" if delta > 0 else "red"
        print(colored(
            f"Size {prev_size} → {curr_size} ({curr_size-prev_size:+d} examples): {delta_str}", 
            color
        ))

def main():
    parser = argparse.ArgumentParser(description="Analyze trained models results")
    parser.add_argument(
        "--models-file",
        type=str,
        default="../res/data/trained_models/trained_models.json",
        help="Path to trained models JSON file"
    )
    args = parser.parse_args()

    # Load models data
    try:
        models = load_trained_models(args.models_file)
    except FileNotFoundError:
        print(colored(f"Error: Could not find models file at {args.models_file}", "red"))
        return
    except json.JSONDecodeError:
        print(colored("Error: Invalid JSON file", "red"))
        return

    # Get available dataset sizes
    dataset_sizes = get_available_dataset_sizes(models)
    size_counts = count_models_by_size(models)
    
    print(colored("\n=== Dataset Statistics ===", "yellow"))
    for i, size in enumerate(dataset_sizes, 1):
        stats = get_size_statistics(models, size)
        print(f"{i}. Size: {size} ({stats['num_models']} models)")
        print(f"   Success rates - Max: {stats['max_rate']:.2%}, Avg: {stats['avg_rate']:.2%}, Min: {stats['min_rate']:.2%}")
    
    analyze_performance_trend(models)
    
    print(colored("\n=== Model Selection ===", "yellow"))
    
    # Get user input
    while True:
        try:
            choice = int(input("\nSelect a dataset size (enter the number): "))
            if 1 <= choice <= len(dataset_sizes):
                selected_size = dataset_sizes[choice - 1]
                break
            print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Please enter a valid number.")

    # Find and display best model for selected size
    best_model = find_best_model_for_size(models, selected_size)
    if best_model:
        print_model_params(best_model)
    else:
        print(colored(f"No models found for dataset size {selected_size}", "red"))

if __name__ == "__main__":
    main()
