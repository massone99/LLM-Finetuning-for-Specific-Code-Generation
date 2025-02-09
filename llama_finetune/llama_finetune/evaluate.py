from datetime import datetime
import re
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from logger import file_logger
# from logger import file_logger

import requests
from typing import Tuple
import json

import sys
import os

# Add the parent directory to sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


nltk.download("punkt")
nltk.download("punkt_tab")

# TODO: extract to separate file
class BuildCheckerClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def process_dataset_inline_content(
        self, data_content, build: bool = False, run: bool = True
    ) -> Tuple[int, int]:
        """
        Process a dataset directly from memory via the inline endpoint.
        """
        payload = {
            "data": data_content,
            "build": build,
            "run": run,
            "use_hashes": False,
        }
        file_logger.write_and_print("Sending request to build checker API inline content\n")

        response = requests.post(f"{self.base_url}/process-dataset-inline", json=payload)
        if response.status_code != 200:
            file_logger.write_and_print(
                f"API Error ({response.status_code}): {response.text}"
            )
            return 0, 0

        result = response.json()
        return result["successful_runs"], result["total_snippets"]


def compute_bleu(reference, candidate):
    """Compute BLEU score between reference and candidate code."""
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        [reference_tokens], candidate_tokens, smoothing_function=smoothie
    )


def generate_code(
    model, tokenizer, prompts, max_new_tokens=512, temperature=1.5, min_p=0.1
):
    """
    Generate code outputs for a list of prompts.
    Args:
        model: The fine-tuned LLM model for code generation.
        tokenizer: Tokenizer associated with the model.
        prompts: List of input prompts to generate code for.
        max_new_tokens: Maximum number of tokens to generate (default: 512).
        temperature: Controls randomness in generation - higher values mean more diverse outputs (default: 1.5).
        min_p: Minimum probability threshold for sampling tokens (default: 0.1).

    Returns:
        List of generated code outputs corresponding to each input prompt.
    """

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
        generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generated_codes.append(generated_code)
    return generated_codes


def compute_bleu_for_model(
    model, tokenizer, test_dataset_path, train_size, output_prefix="baseline"
) -> Tuple[str, float]:
    """
    Evaluate model performance using BLEU metric.
    
    Returns:
        Tuple[str, float]: A tuple containing the output file path and the average BLEU score
    """
    from evaluate import compute_bleu
    from datasets import load_dataset
    from datetime import datetime

    # Load test dataset
    test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")
    test_df = test_dataset.to_pandas()

    # Extract prompts and references
    def extract_conversations(convo_list):
        try:
            human_msg = convo_list[0]["value"]
            assistant_msg = convo_list[1]["value"]
            return pd.Series({"prompt": human_msg, "reference": assistant_msg})
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error extracting conversations: {e}")
            return pd.Series({"prompt": None, "reference": None})

    test_df[["prompt", "reference"]] = test_df["conversations"].apply(
        extract_conversations
    )
    test_df.dropna(subset=["prompt", "reference"], inplace=True)

    # Generate code
    test_prompts = test_df["prompt"].tolist()
    test_references = test_df["reference"].tolist()
    generated_codes = generate_code(model, tokenizer, test_prompts)

    # Calculate metrics
    bleu_scores = []
    results = []

    # Calculate individual BLEU scores
    for ref, gen in zip(test_references, generated_codes):
        bleu = compute_bleu(ref, gen)
        bleu_scores.append(bleu)

        results.append({"reference": ref, "generated": gen, "bleu": bleu})

    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    avg_metrics = {"bleu": avg_bleu}

    # Prepare evaluation results
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "average_metrics": avg_metrics,
        "detailed_results": results,
        "training_dataset_size": train_size,
    }

    # Create results directory if it doesn't exist
    os.makedirs("../res/data/results", exist_ok=True)

    # Save evaluation results
    results_file = f'../res/data/results/evaluation_results_{output_prefix}_{datetime.now().strftime("%Y%m%d")}.json'
    with open(results_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    eval_res_title_str = f"\nEvaluation Results ({output_prefix}):\n"
    bleu_score_str = f"\nAverage BLEU Score: {avg_bleu:.4f}"
    detailed_res_str = f"\nDetailed results saved to: {results_file}"

    file_logger.write_and_print(eval_res_title_str, heading=2)
    file_logger.write_and_print(bleu_score_str, heading=3)
    file_logger.write_and_print(detailed_res_str)

    return results_file, avg_bleu


def extract_generated_code(output_file, output_prefix) -> Tuple[bool, str]:
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
    from retrieve_model_output import process_evaluation_results

    # Extract and save generated code
    generated_code_dir = os.path.join("../res/data/generated_code", output_prefix)
    success, message, files = process_evaluation_results(
        output_file, generated_code_dir, file_extension=".txt"
    )
    if success:
        print(f"Generated code samples saved to: {generated_code_dir}\n")
    else:
        print(f"Warning: Failed to extract code samples: {message}")
    return success, generated_code_dir if success else ""



def convert_pairs_to_json(folder_path):
    # This function scans the folder for files named prompt_X.txt and code_X.txt,
    # reads their content, and constructs a JSON structure with "human" and "assistant" messages.

    # Regex to capture indices from filenames like prompt_1.txt or code_12.txt
    file_pattern = re.compile(r'^(prompt|code)_(\d+)\.txt$')
    # Store found prompts and codes using the index as key
    prompts = {}
    codes = {}

    for file_name in os.listdir(folder_path):
        match = file_pattern.match(file_name)
        if match:
            file_type = match.group(1)    # "prompt" or "code"
            idx = match.group(2)         # e.g. "1", "12"
            full_path = os.path.join(folder_path, file_name)

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_type == "prompt":
                prompts[idx] = content.strip()
            else:
                codes[idx] = content.strip()

    # Build the final list of conversation objects
    result = []
    # Sort indices to keep final JSON in ascending order
    all_indices = sorted(set(prompts.keys()).union(codes.keys()), key=int)

    for idx in all_indices:
        prompt_text = prompts.get(idx, "")
        code_text = codes.get(idx, "")
        if prompt_text or code_text:
            conv = {
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt_text
                    },
                    {
                        "from": "assistant",
                        "value": code_text
                    }
                ]
            }
            result.append(conv)

    return json.dumps(result, indent=2)

def evaluate_model(
    model, tokenizer, test_dataset_path, train_size, output_prefix="baseline"
):
    '''
    Evaluate the model using the specified test dataset and training size.
    
    Args:
        model: The fine-tuned LLM model for code generation.
        tokenizer: Tokenizer associated with the model.
        test_dataset_path: The path to the test dataset.
        train_size: The size of the training dataset.
        output_prefix: The prefix to use for the output files.
    
    Returns:
        Tuple[str, float, Tuple[int, int]]: A tuple containing the output file path, the average BLEU score, 
            and a tuple containing the number of successful runs and the total number of snippets processed.
    '''
    dataset_path = None
    try:
        dataset_path, avg_bleu = compute_bleu_for_model(
            model, tokenizer, test_dataset_path, train_size, output_prefix
        )
        if dataset_path is None:
            file_logger.write_and_print("Error: Failed to compute BLEU scores.")
            raise Exception("Failed to compute BLEU scores.")
        
        success, generated_code_dir = extract_generated_code(dataset_path, output_prefix)
        if not success or not generated_code_dir:
            file_logger.write_and_print("Error: Failed to extract generated code.")
            raise Exception("Failed to extract generated code.")
        
        # Instead of saving to file and reloading, process directly:
        mapped_dataset = convert_pairs_to_json(generated_code_dir)
        if not mapped_dataset:
            file_logger.write_and_print("Error: Failed to map generated code to dataset format.")
            raise Exception("Failed to map generated code to dataset format.")
        
        # Convert mapped_dataset (string) to Python object
        data_for_checker = json.loads(mapped_dataset)

        print("Processing dataset inline...")
        client = BuildCheckerClient()
        work_sampl, tot_sampl = client.process_dataset_inline_content(data_for_checker, run=True)
        file_logger.write_and_print(f"\nRunning examples: {work_sampl}/{tot_sampl}\n")
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        raise e

    return dataset_path, avg_bleu, (work_sampl, tot_sampl)