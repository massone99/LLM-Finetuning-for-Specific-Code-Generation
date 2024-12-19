import tkinter as tk
from tkinter import filedialog
import hashlib
import json
import os
import sys

# Add parent directory to path to import retrieve_model_output
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from build_checker.build_checker.log.logger import logger


hash_file_path = "res/config/processed_hashes.json"
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scala_proj_dir = current_dir + "/res/akka_placeholder"

def load_processed_hashes(hash_file_path):
    if os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as f:
            processed_hashes = set(json.load(f))
        logger.info(f"Loaded {len(processed_hashes)} processed hashes.")
    else:
        processed_hashes = set()
        logger.info("No processed hashes file found. Starting fresh.")
    return processed_hashes


def select_file() -> str:
    """Opens a file dialog to select a file.

    Returns:
        str: The path of the selected file.
    """
    file_path = filedialog.askopenfilename(
        title="Select a File",
        initialdir="../dataset_builder/data/",  # Starting directory set to current working directory
        filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
    )
    if file_path:
        logger.info(f"Selected file: {file_path}")
        return file_path
    else:
        logger.warning("No file selected.")


def load_json_dataset(json_file_path) -> dict:
    if not os.path.exists(json_file_path):
        logger.error(f"The file '{json_file_path}' does not exist.")
        return None

    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded dataset from '{json_file_path}'.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON file: {e}")
        return None

def compute_hash(prompt, code):
    concatenated = (prompt + code).encode('utf-8')
    return hashlib.sha256(concatenated).hexdigest()

def save_processed_hashes(processed_hashes, hash_file_path):
    with open(hash_file_path, "w") as f:
        json.dump(list(processed_hashes), f)
    logger.info(f"Saved {len(processed_hashes)} processed hashes.")


def process_projects(dataset, output_directory, build_flag, run_flag):
    if dataset is None:
        logger.error("No dataset file provided. Stopping further processing.")
        return
    
    processed_hashes = load_processed_hashes(hash_file_path)
    
    # Iterate over each conversation
    for idx, conversation in enumerate(dataset):
        assistant_messages = [
            msg["value"]
            for msg in conversation.get("conversations", [])
            if msg.get("from") == "assistant"
        ]
        
        human_prompts = [
            msg["value"]
            for msg in conversation.get("conversations", [])
            if msg.get("from") == "human"
        ]

        for code, human_prompt in zip(assistant_messages, human_prompts):
            current_hash = compute_hash(human_prompt, code)
            if current_hash in processed_hashes:
                logger.info(f"Skipping already processed pair at index {idx}.")
                continue
            
            # Define the path to the Main.scala file
            main_scala_path = os.path.join(
                output_directory, "src/main/scala/Main.scala"
            )
            logger.debug("Path of scala file: " + main_scala_path)

            with open(main_scala_path, "w") as scala_file:
                scala_file.write(code)

            logger.info(f"Wrote code to {main_scala_path}")

            # Build the Scala project
            if build_flag:
                build_project(output_directory)

            # Run the Scala project
            if run_flag:
                run_status = run_project(output_directory)
                if not run_status:
                    logger.error("Error running project. Prompt and code returning error:\n")
                    logger.error(f"Idx conversation: {idx}")
                    logger.error(f"Prompt: {human_prompt}")
                    # logger.error(f"Code: {code}")
                    
                    return
            processed_hashes.add(current_hash)
            save_processed_hashes(processed_hashes, hash_file_path)
            logger.info(f"Conversation idx: {idx+1} processed successfully.")

    logger.info(f"{len(dataset)} conversations processed successfully.")

def run_project(project_directory) -> bool:
    import subprocess

    logger.debug(f"Running project in directory: {project_directory}")
    try:
        result = subprocess.run(
            ["sbt", "run"],
            cwd=project_directory,
            # If the command launch fails, an exception is raised
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        return False
    except FileNotFoundError:
        logger.error("sbt not found. Please ensure sbt is installed and in your PATH")
        return False

def build_project(project_directory):
    import subprocess

    logger.debug(f"Building project in directory: {project_directory}")
    try:
        result = subprocess.run(
            ["sbt", ""],
            cwd=project_directory,
            # If the command launch fails, an exception is raised
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Build stdout: {result.stdout.decode('utf-8')}")
        logger.error(f"Build stderr: {result.stderr.decode('utf-8')}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error building project: {e}")
    except FileNotFoundError:
        logger.error("sbt not found. Please ensure sbt is installed and in your PATH")
        return False
    

def __calc_working_code_samples(dataset, run_flag) -> tuple:
    from retrieve_model_output import extract_prompt_and_code
    from retrieve_model_output import extract_prompt_and_code_qwen
    
    if dataset is None:
        logger.error("No dataset file provided. Stopping further processing.")
        return

    running_examples = 0
    
    results = dataset["detailed_results"]
    dataset_len = dataset["training_dataset_size"]
    for chat in results:
        generated = chat["generated"]
        prompt, code = extract_prompt_and_code(generated)
        
        # Define the path to the Main.scala file
        main_scala_path = os.path.join(
            scala_proj_dir, "src/main/scala/Main.scala"
        )
        
        with open(main_scala_path, "w") as scala_file:
            scala_file.write(code)
        
        # Run the Scala project
        if run_flag:
            run_status = run_project(scala_proj_dir)
            if not run_status:
                logger.error("Code causing error!")
            else:
                running_examples += 1
    logger.debug(f"Running examples: {running_examples}/{len(results)}")
    return running_examples, len(results)
    

def evaluate_generated_code(dataset_path, run_flag) -> tuple:
    """
    Evaluates the code generated by the finetuned model based on the flags provided.

    Args:
        dataset_path (str): path of the dataset containing the generated code
        scala_dir (str): path to the Scala project directory where the code should be written
        build_flag (bool): flag indicating whether to build the project
        run_flag (bool): flag indicating whether to run the project

    Raises:
        ValueError: if no dataset file is provided

    Returns:
        tuple: a tuple containing the number of running examples and the total number of examples
    """
    dataset = load_json_dataset(dataset_path)
    if dataset is None:
        logger.error("No dataset file provided. Stopping further processing.")
        raise ValueError("No dataset file provided.")
    
    return __calc_working_code_samples(dataset, run_flag)



def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # selected_file_path = select_file()
    selected_file_path = "../dataset_builder/data/synthetic_data/dataset_llama.json"

    # FIXME
    dataset = load_json_dataset(selected_file_path)
    output_path = "res/akka_placeholder"
    process_projects(dataset, output_path, build_flag=False, run_flag=True)

    # Loading results JSON
    # selected_file_path = "/home/lorix/Documents/dev/uni/TESI/python/llama_finetune/res/data/results/evaluation_results_loaded_finetuned_trainsize43_20241216.json"
    
    # dataset = load_json_dataset(selected_file_path)
    # __calc_working_code_samples(dataset, run_flag=True)
    
    