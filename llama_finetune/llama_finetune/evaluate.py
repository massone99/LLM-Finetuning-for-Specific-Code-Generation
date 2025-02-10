from datetime import datetime
import json
import os
import re
import sys
from typing import Tuple, List, Dict, Any

import nltk
import pandas as pd
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from build_client import BuildCheckerClient
from logger import file_logger

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


class CodeGenerator:
    @staticmethod
    def generate_code(
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 1.5,
        min_p: float = 0.1,
    ) -> List[str]:
        """Generate code outputs for a list of prompts."""
        FastLanguageModel.for_inference(model)
        generated_codes = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                min_p=min_p,
            )
            generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)[
                0
            ]
            generated_codes.append(generated_code)

        return generated_codes


class MetricsCalculator:
    @staticmethod
    def compute_bleu(reference: str, candidate: str) -> float:
        """Compute BLEU score between reference and candidate code."""
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        smoothie = SmoothingFunction().method4
        return sentence_bleu(
            [reference_tokens], candidate_tokens, smoothing_function=smoothie
        )

    @staticmethod
    def calculate_metrics(
        references: List[str], generated_codes: List[str]
    ) -> Tuple[List[Dict], float]:
        """Calculate BLEU scores for each pair and average."""
        results = []
        bleu_scores = []

        for ref, gen in zip(references, generated_codes):
            bleu = MetricsCalculator.compute_bleu(ref, gen)
            bleu_scores.append(bleu)
            results.append({"reference": ref, "generated": gen, "bleu": bleu})

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        return results, avg_bleu


class DataProcessor:
    @staticmethod
    def extract_conversations(convo_list: List[Dict]) -> pd.Series:
        """Extract human and assistant messages from conversation list."""
        try:
            human_msg = convo_list[0]["value"]
            assistant_msg = convo_list[1]["value"]
            return pd.Series({"prompt": human_msg, "reference": assistant_msg})
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error extracting conversations: {e}")
            return pd.Series({"prompt": None, "reference": None})

    @staticmethod
    def convert_pairs_to_json(folder_path: str) -> str:
        """Convert prompt-code pairs to JSON format."""
        file_pattern = re.compile(r"^(prompt|code)_(\d+)\.txt$")
        prompts = {}
        codes = {}

        for file_name in os.listdir(folder_path):
            match = file_pattern.match(file_name)
            if match:
                file_type, idx = match.group(1), match.group(2)
                with open(
                    os.path.join(folder_path, file_name), "r", encoding="utf-8"
                ) as f:
                    content = f.read().strip()

                if file_type == "prompt":
                    prompts[idx] = content
                else:
                    codes[idx] = content

        result = []
        all_indices = sorted(set(prompts.keys()).union(codes.keys()), key=int)

        for idx in all_indices:
            prompt_text = prompts.get(idx, "")
            code_text = codes.get(idx, "")
            if prompt_text or code_text:
                result.append(
                    {
                        "conversations": [
                            {"from": "human", "value": prompt_text},
                            {"from": "assistant", "value": code_text},
                        ]
                    }
                )

        return json.dumps(result, indent=2)


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
