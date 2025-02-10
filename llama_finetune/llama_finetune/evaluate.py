from datetime import datetime
import json
import os
import sys
from typing import Tuple, List, Dict, Any

import nltk
import pandas as pd
from datasets import load_dataset
from evaluation_utils.build_client import BuildCheckerClient
from logger import file_logger

# Import the classes from their new files
from evaluation_utils.code_generator import CodeGenerator
from evaluation_utils.metrics_calculator import MetricsCalculator
from evaluation_utils.data_processor import DataProcessor

# Constants
RESULTS_DIR = "../res/data/results"
GENERATED_CODE_DIR = "../res/data/generated_code"

# Setup
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from retrieve_model_output import process_evaluation_results

nltk.download("punkt")
nltk.download("punkt_tab")

def compute_bleu_for_model(
    model, tokenizer, test_dataset_path, train_size, output_prefix="baseline"
) -> Tuple[str, float]:
    """Evaluate model performance using BLEU metric."""
    # Load and prepare test data
    test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")
    test_df = test_dataset.to_pandas()

    test_df[["prompt", "reference"]] = test_df["conversations"].apply(
        DataProcessor.extract_conversations
    )
    test_df.dropna(subset=["prompt", "reference"], inplace=True)

    # Generate and evaluate code
    test_prompts = test_df["prompt"].tolist()
    test_references = test_df["reference"].tolist()
    generated_codes = CodeGenerator.generate_code(model, tokenizer, test_prompts)

    results, avg_bleu = MetricsCalculator.calculate_metrics(
        test_references, generated_codes
    )

    # Save results
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "average_metrics": {"bleu": avg_bleu},
        "detailed_results": results,
        "training_dataset_size": train_size,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(
        RESULTS_DIR,
        f'evaluation_results_{output_prefix}_{datetime.now().strftime("%Y%m%d")}.json',
    )

    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    # Log results
    file_logger.write_and_print(f"\nEvaluation Results ({output_prefix}):\n", heading=2)
    file_logger.write_and_print(f"\nAverage BLEU Score: {avg_bleu:.4f}", heading=3)
    file_logger.write_and_print(f"\nDetailed results saved to: {results_file}")

    return results_file, avg_bleu


def extract_generated_code(output_file: str, output_prefix: str) -> Tuple[bool, str]:
    """
    Extracts generated code from the specified output file and saves it to a directory.

    Args:
        output_file (str): The path to the file containing the generated code output.
        output_prefix (str): The prefix to use for the directory where the generated code will be saved.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating success and the directory where the code samples were saved.

    Raises:
        Any exceptions raised by `process_evaluation_results` will propagate.
    """
    # Extract and save generated code
    generated_code_dir = os.path.join(GENERATED_CODE_DIR, output_prefix)
    success, message, files = process_evaluation_results(
        output_file, generated_code_dir, file_extension=".txt"
    )
    if success:
        file_logger.write_and_print(
            f"Generated code samples saved to: {generated_code_dir}\n"
        )
    else:
        file_logger.write_and_print(
            f"Warning: Failed to extract code samples: {message}"
        )
    return success, generated_code_dir if success else ""


def evaluate_model(
    model: Any,
    tokenizer: Any,
    test_dataset_path: str,
    train_size: int,
    output_prefix: str = "baseline",
) -> Tuple[str, float, Tuple[int, int]]:
    """
    Evaluate the model using the specified test dataset and training size.

    Args:
        model: The fine-tuned LLM model for code generation.
        tokenizer: Tokenizer associated with the model.
        test_dataset_path: The path to the test dataset.
        train_size: The size of the training dataset.
        output_prefix: The prefix to use for the output files.

    Returns:
        Tuple[str, float, Tuple[int, int]]: A tuple containing:
            - output file path (str)
            - average BLEU score (float)
            - tuple of (successful runs, total snippets)

    Raises:
        Exception: If any step of the evaluation process fails
    """
    try:
        # Step 1: Compute BLEU scores
        dataset_path, avg_bleu = compute_bleu_for_model(
            model, tokenizer, test_dataset_path, train_size, output_prefix
        )
        if dataset_path is None:
            raise ValueError("Failed to compute BLEU scores")

        # Step 2: Extract generated code
        success, generated_code_dir = extract_generated_code(
            dataset_path, output_prefix
        )
        if not success or not generated_code_dir:
            raise ValueError("Failed to extract generated code")

        # Step 3: Process the generated code
        mapped_dataset = DataProcessor.convert_pairs_to_json(generated_code_dir)
        if not mapped_dataset:
            raise ValueError("Failed to map generated code to dataset format")

        # Step 4: Run code validation
        file_logger.write_and_print("Processing dataset inline...", heading=3)
        client = BuildCheckerClient()
        work_sampl, tot_sampl = client.process_dataset_inline_content(
            json.loads(mapped_dataset), run=True
        )

        file_logger.write_and_print(
            f"\nRunning examples: {work_sampl}/{tot_sampl}\n", heading=3
        )

        return dataset_path, avg_bleu, (work_sampl, tot_sampl)

    except Exception as e:
        file_logger.write_and_print(f"Error during model evaluation: {str(e)}")
        raise
