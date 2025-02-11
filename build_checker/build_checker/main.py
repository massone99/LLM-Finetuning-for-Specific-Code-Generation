import hashlib
import json
import os
import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, 
    QCheckBox, QApplication, QFileDialog, QTextEdit
)

# Configure logging
logger = logging.getLogger("BuildCheckerLogger")
logger.setLevel(logging.INFO)

# Create console handler with custom formatter
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Prevent log propagation to avoid duplicate logs
logger.propagate = False

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from api import BuildCheckerAPI

hash_file_path = "res/config/processed_hashes.json"
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scala_proj_dir = current_dir + "/res/akka_placeholder"

project_dir = str(ROOT_DIR) + "/build_checker"

output_directory =  project_dir + "/res/akka_placeholder"
# Define the path to the Main.scala file
main_scala_path = os.path.join(output_directory, "src/main/scala/Main.scala")
failing_snippets_path =  project_dir + "/res/config/failing_snippets.json"

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
    """Opens a file dialog to select a file using PyQt5.
    
    Returns:
        str: The path of the selected file.
    """
    dialog = QFileDialog()
    file_path, _ = dialog.getOpenFileName(
        caption="Select Dataset File",
        directory="../dataset_builder/data/",
        filter="JSON Files (*.json);;All Files (*.*)"
    )
    if file_path:
        logger.info(f"Selected file: {file_path}")
        return file_path
    else:
        logger.warning("No file selected.")
        return None

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
    concatenated = (prompt + code).encode("utf-8")
    return hashlib.sha256(concatenated).hexdigest()

def save_processed_hashes(processed_hashes, hash_file_path):
    with open(hash_file_path, "w") as f:
        json.dump(list(processed_hashes), f)
    logger.info(f"Saved {len(processed_hashes)} processed hashes.")

def save_failing_snippets(failing_snippets, failing_snippets_path):
    with open(failing_snippets_path, "w") as f:
        json.dump(failing_snippets, f, indent=4)
    logger.info(f"Saved {len(failing_snippets)} failing snippets.")

def process_snippets(dataset, build_flag, run_flag, use_hashes=False, on_working_snippet=None):
    if dataset is None:
        logger.error("No dataset file provided. Stopping further processing.")
        return

    processed_hashes = load_processed_hashes(hash_file_path) if use_hashes else set()
    successful_runs = 0
    total_snippets = 0
    failing_snippets = []
    api = BuildCheckerAPI()

    # Iterate over each conversation
    for idx, conversation in enumerate(dataset):
        assistant_messages, human_prompts = api._get_prompt_and_code(conversation)

        for code, human_prompt in zip(assistant_messages, human_prompts):
            total_snippets += 1
            current_hash = compute_hash(human_prompt, code)
            if use_hashes and current_hash in processed_hashes:
                continue

            success, error_msg = api.test_single_snippet(code, build_flag, run_flag, idx=idx, prompt=human_prompt)
            
            if success:
                successful_runs += 1
                if on_working_snippet:
                    on_working_snippet(idx, human_prompt, code)
            else:
                failing_snippets.append({
                    "idx": idx,
                    "prompt": human_prompt,
                    "code": code,
                    "error_output": error_msg,
                    "error_cause": error_msg.split('\n')[-1] if error_msg else "Unknown error"
                })

            if use_hashes:
                processed_hashes.add(current_hash)
                save_processed_hashes(processed_hashes, hash_file_path)

    save_failing_snippets(failing_snippets, failing_snippets_path)
    logger.info(f"{successful_runs}/{total_snippets} snippets ran successfully.")
    logger.info(f"{len(dataset)} conversations processed successfully.")

def get_prompt_and_code(conversation):
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
    return assistant_messages, human_prompts

def run_project(project_directory) -> tuple:
    import subprocess

    logger.debug(f"Running project in directory: {project_directory}")
    try:
        result = subprocess.run(
            ["sbt", "run"],
            cwd=project_directory,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Use text mode for string output
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        # Include both stderr and stdout in error output for better debugging
        error_output = f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        return False, error_output
    except FileNotFoundError:
        logger.error("sbt not found. Please ensure sbt is installed and in your PATH")
        return False, "sbt not found. Please ensure sbt is installed and in your PATH"

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
    """
    Calculate the number of working code samples in the dataset
    """
    from retrieve_model_output import extract_prompt_and_code

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
        main_scala_path = os.path.join(scala_proj_dir, "src/main/scala/Main.scala")

        with open(main_scala_path, "w") as scala_file:
            scala_file.write(code)

        # Run the Scala project
        if run_flag:
            run_status, run_output = run_project(scala_proj_dir)
            if not run_status:
                logger.error("Code causing error!")
                logger.error(f"Run output: {run_output}")
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

class TestSnippetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test Snippet")
        self.setGeometry(200, 200, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add text edit for code input
        self.code_input = QTextEdit()
        self.code_input.setPlaceholderText("Paste your Scala code here...")
        
        # Add test button
        self.test_button = QPushButton("Test Snippet")
        self.test_button.clicked.connect(self.test_snippet)
        
        # Add result label
        self.result_label = QLabel("")
        
        # Add widgets to layout
        layout.addWidget(self.code_input)
        layout.addWidget(self.test_button)
        layout.addWidget(self.result_label)

    def test_snippet(self):
        code = self.code_input.toPlainText()
        if not code.strip():
            self.result_label.setText("Please enter some code")
            return

        # Format and de-minify the code
        formatted_code = self.format_scala_code(code)

        # Write code to scala file
        with open(main_scala_path, "w") as scala_file:
            scala_file.write(formatted_code)

        # Run the project
        run_status, run_output = run_project(output_directory)
        
        if run_status:
            self.result_label.setText("Success! Code ran without errors.")
            logger.info("Success! Code ran without errors.")  # Print to stdout
        else:
            error_lines = run_output.strip().split('\n')
            error_cause = error_lines[-1] if error_lines else "Unknown error"
            self.result_label.setText(f"Error: {error_cause}")
            logger.error(f"Error: {error_cause}")  # Print to stdout
            logger.error(f"Run output: {run_output}") # Print full output to stdout

    def format_scala_code(self, code):
        formatted_code = code.replace("\\n", "\n").replace("\\t", "\t")
        return formatted_code # Print full output to stdout

class DatasetProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Processor")
        self.setGeometry(100, 100, 600, 200)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # File selection area
        self.file_label = QLabel("No file selected")
        self.select_button = QPushButton("Select Dataset")
        self.select_button.clicked.connect(self.select_file)

        # Checkboxes for flags
        self.build_checkbox = QCheckBox("Build Project")
        self.run_checkbox = QCheckBox("Run Project")
        self.run_checkbox.setChecked(True)  # Default to True

        # Add hash checkbox
        self.hash_checkbox = QCheckBox("Use Processed Hashes")
        self.hash_checkbox.setChecked(False)

        # Add append checkbox
        self.append_checkbox = QCheckBox("Append working snippets to main dataset")
        self.append_checkbox.setChecked(False)
        self.append_checkbox.stateChanged.connect(self.handle_append_state)

        # Add dataset path label
        self.main_dataset_label = QLabel("Main dataset: Not selected")
        self.main_dataset_label.setVisible(False)

        # Add select main dataset button
        self.select_main_dataset_button = QPushButton("Select Main Dataset")
        self.select_main_dataset_button.clicked.connect(self.select_main_dataset)
        self.select_main_dataset_button.setVisible(False)

        # Process button
        self.process_button = QPushButton("Process Dataset")
        self.process_button.clicked.connect(self.process_dataset)
        self.process_button.setEnabled(False)

        # Status label
        self.status_label = QLabel("")

        # Add widgets to layout
        layout.addWidget(self.file_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.build_checkbox)
        layout.addWidget(self.run_checkbox)
        layout.addWidget(self.hash_checkbox)
        layout.addWidget(self.append_checkbox)
        layout.addWidget(self.main_dataset_label)
        layout.addWidget(self.select_main_dataset_button)
        layout.addWidget(self.process_button)
        layout.addWidget(self.status_label)

        self.selected_file = None
        self.main_dataset_path = None
        
        # Add test snippet button
        self.test_button = QPushButton("Test Snippet")
        self.test_button.clicked.connect(self.open_test_window)
        
        # Add to layout (add before status_label)
        layout.addWidget(self.test_button)
        layout.addWidget(self.status_label)
        
        # Add test window property
        self.test_window = None
    
    def open_test_window(self):
        if not self.test_window:
            self.test_window = TestSnippetWindow()
        self.test_window.show()

    def handle_append_state(self, state):
        self.main_dataset_label.setVisible(state)
        self.select_main_dataset_button.setVisible(state)
        if not state:
            self.main_dataset_path = None
            self.main_dataset_label.setText("Main dataset: Not selected")

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            "../dataset_builder/data/",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if file_path:
            self.selected_file = file_path
            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.process_button.setEnabled(True)
            self.hash_checkbox.setEnabled(True)

    def select_main_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Main Dataset File",
            str(ROOT_DIR),
            "JSON Files (*.json);;All Files (*.*)"
        )
        if file_path:
            self.main_dataset_path = file_path
            self.main_dataset_label.setText(f"Main dataset: {os.path.basename(file_path)}")

    def is_duplicate_snippet(self, prompt: str, code: str, main_dataset: list) -> bool:
        """Check if either prompt or code already exists in the main dataset"""
        for entry in main_dataset:
            for conversation in entry.get("conversations", []):
                if conversation.get("value") in [prompt, code]:
                    return True
        return False

    def process_dataset(self):
        if self.selected_file:
            self.status_label.setText("Loading dataset...")
            dataset = load_json_dataset(self.selected_file)

            if dataset:
                self.status_label.setText("Processing dataset...")
                try:
                    working_snippets = []
                    
                    # Load main dataset early if we need to check for duplicates
                    main_dataset = None
                    if self.append_checkbox.isChecked() and self.main_dataset_path:
                        with open(self.main_dataset_path, 'r') as f:
                            main_dataset = json.load(f)
                    
                    def append_snippet_to_main_dataset(idx, prompt, code):
                        if main_dataset and not self.is_duplicate_snippet(prompt, code, main_dataset):
                            logger.info(f"Appending working snippet n°{idx} to main dataset: {prompt[:10]}...")
                            working_snippets.append({
                                "conversations": [
                                    {"from": "human", "value": prompt},
                                    {"from": "assistant", "value": code}
                                ]
                            })
                        else:
                            logger.info(f"Skipping duplicate snippet n°{idx}")

                    process_snippets(
                        dataset,
                        build_flag=self.build_checkbox.isChecked(),
                        run_flag=self.run_checkbox.isChecked(),
                        use_hashes=self.hash_checkbox.isChecked(),
                        on_working_snippet=append_snippet_to_main_dataset if self.append_checkbox.isChecked() else None
                    )

                    if self.append_checkbox.isChecked() and working_snippets and main_dataset:
                        # Append working snippets
                        main_dataset.extend(working_snippets)
                        
                        # Save updated dataset
                        with open(self.main_dataset_path, 'w') as f:
                            json.dump(main_dataset, f, indent=2)
                        
                        self.status_label.setText(
                            f"Processing completed! Added {len(working_snippets)} new working snippets to main dataset"
                        )
                    else:
                        self.status_label.setText("Processing completed successfully!")
                except Exception as e:
                    self.status_label.setText(f"Error during processing: {str(e)}")
                    raise e
            else:
                self.status_label.setText("Error loading dataset")

def main():
    app = QApplication(sys.argv)
    window = DatasetProcessorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
