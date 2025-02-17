import os
from datasets import load_dataset
from peft import PeftModel
from transformers import TrainingArguments
from PyQt5.QtWidgets import QApplication, QFileDialog

import json
from datetime import datetime

from training.config import PEFT_PARAMS, TRAINING_PARAMS, store_model_info

from training.model_loader import load_model_and_tokenizer
from training.grid_search import execute_smac_optimization
from training.training_utils import process_trained_model


def show_dataset_dialog():
    """Show a file dialog to select the dataset file"""
    app = QApplication([])
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle("Select Dataset File")
    file_dialog.setNameFilter("JSON files (*.json)")
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    
    if file_dialog.exec_():
        selected_files = file_dialog.selectedFiles()
        return selected_files[0]
    return None

def show_folder_dialog():
    """Show a folder dialog to select a directory"""
    app = QApplication([])
    folder = QFileDialog.getExistingDirectory(None, "Select Model Folder", os.getcwd())
    return folder if folder else None

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
        default=None,
        help="Path to training dataset",
    )
    parser.add_argument(
        "--gui-select",
        action="store_true",
        help="Select dataset file using GUI dialog",
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
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials for SMAC optimization",
    )
    return parser.parse_args()


def process_loaded_model(
    args, model, output_dir, tokenizer, train_dataset_size
) -> None:
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
        PEFT_PARAMS,
    )


def save_grid_search_results(output_dir: str, best_params: dict) -> None:
    """Save grid search results to a JSON file."""
    if not best_params:
        return

    print("\nBest performing combination:")
    print(
        f"Execution rate: {best_params['execution_rate']:.2%} "
        f"({best_params['running_snippets']}/{best_params['total_snippets']})"
    )
    print(f"BLEU score: {best_params['bleu']:.4f}")
    print(f"PEFT parameters: {best_params['peft']}")
    print(f"Training parameters: {best_params['training']}")

    results_dir = os.path.join(output_dir, "grid_search_results")
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(
        results_dir, f'best_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

    with open(results_file, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest parameters saved to: {results_file}")


# Training normale
# poetry run python train.py --dataset-path ../res/data/dataset_llama.json --output-dir ../res/outputs

# To select the dataset with a GUI, append the flag --gui-select

# Grid search (with Smac) (TODO: FIX)
# poetry run python train.py --dataset-path ../res/data/dataset_llama.json --output-dir ../res/outputs --grid-search


# Caricamento di un modello esistente
# poetry run python train.py --load-model ../res/outputs/finetuned_model
def main():
    # Verify working directory
    current_dir = os.path.basename(os.getcwd())
    parent_dir = os.path.basename(os.path.dirname(os.getcwd()))
    if not (current_dir == "llama_finetune" and parent_dir == "llama_finetune"):
        raise RuntimeError(
            "This script must be run from inside the llama_finetune/llama_finetune directory"
        )

    print("Current working directory: CORRECT")

    args = parse_args()
    print("Args parsed!")

    # Handle dataset selection
    if args.gui_select:
        dataset_path = show_dataset_dialog()
        if dataset_path is None:
            print("No dataset selected. Exiting...")
            return
        args.dataset_path = dataset_path
    elif args.dataset_path is None:
        args.dataset_path = "../res/data/dataset_llama.json"  # default path

    print(f"Using dataset: {args.dataset_path}")

    # Configuration
    max_seq_length = 4096
    output_dir = args.output_dir

    # Load initial model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=max_seq_length
    )

    # Get training dataset size
    try:
        train_dataset = load_dataset(
            "json", data_files=args.dataset_path, split="train"
        )
        train_dataset_size = len(train_dataset)
    except Exception as e:
        print(f"Warning: Could not determine training dataset size: {e}")
        train_dataset_size = None

    try:
        if args.load_model:
            process_loaded_model(args, model, output_dir, tokenizer, train_dataset_size)
        elif args.grid_search:
            best_params = execute_smac_optimization(
                args, model, tokenizer, max_seq_length, train_dataset_size
            )
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
                TRAINING_PARAMS,
            )
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
