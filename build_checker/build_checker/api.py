import os
import json
import hashlib
import subprocess
from pathlib import Path
from log.logger import logger

class BuildCheckerAPI:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_path = os.path.dirname(self.current_dir)
        self.output_directory = root_path + "/build_checker" + "/res/akka_placeholder"
        self.scala_proj_dir = self.current_dir + "/res/akka_placeholder"
        self.hash_file_path = "res/config/processed_hashes.json"
        root_path = Path(self.current_dir).parent
        build_checker_path = root_path / "build_checker"
        akka_project_path = build_checker_path / "res/akka_placeholder"
        self.failing_snippets_path = build_checker_path / "res/config/failing_snippets.json"
        self.main_scala_path = os.path.join(akka_project_path, "src/main/scala/Main.scala")

    def load_json_dataset(self, json_file_path) -> dict:
        if not os.path.exists(json_file_path):
            logger.error(f"The file '{json_file_path}' does not exist.")
            return None
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
                # If the data contains evaluation results, extract the generated code samples
                if isinstance(data, dict) and 'detailed_results' in data:
                    return [
                        {'conversations': [
                            {'from': 'assistant', 'value': result['generated']}
                        ]}
                        for result in data['detailed_results']
                    ]
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON file: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading JSON file: {e}")
            return None

    def process_snippets(self, dataset, build_flag, run_flag, use_hashes=False):
        if not dataset:
            return False, "No dataset provided"

        processed_hashes = self._load_processed_hashes() if use_hashes else set()
        successful_runs = 0
        total_snippets = 0
        failing_snippets = []

        for idx, conversation in enumerate(dataset):
            assistant_msgs, human_prompts = self._get_prompt_and_code(conversation)
            for code, prompt in zip(assistant_msgs, human_prompts):
                success, msg = self.test_single_snippet(code, build_flag, run_flag, idx=idx, prompt=prompt)
                if success:
                    successful_runs += 1
                else:
                    failing_snippets.append({
                        "idx": idx,
                        "prompt": prompt,
                        "code": code,
                        "error": msg
                    })
                total_snippets += 1

        self._save_failing_snippets(failing_snippets)
        return successful_runs, total_snippets

    def test_single_snippet(self, code: str, build=True, run=True, idx=None, prompt=None) -> tuple[bool, str]:
        if not code.strip():
            return False, "No code provided"

        logger.info(f"Processing snippet: {code[:10]}...")

        with open(self.main_scala_path, "w") as f:
            f.write(code)

        if build:
            build_success, build_msg = self.build_project()
            if not build_success:
                if idx is not None and prompt is not None:
                    logger.error(f"Build failed for idx: {idx}")
                    logger.error(f"Prompt: {prompt}")
                return False, f"Build failed: {build_msg}"

        if run:
            success, msg = self.run_project()
            if not success and idx is not None and prompt is not None:
                logger.error(f"Run failed for idx: {idx}")
                logger.error(f"Prompt: {prompt}")
            return success, msg

        return True, "Code written successfully"

    def build_project(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(["sbt", "compile"],
                        cwd=self.output_directory,
                        check=True,
                        capture_output=True,
                        text=True)
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr
            logger.error(f"Build error: {error_msg}")
            return False, error_msg
        except FileNotFoundError as e:
            logger.error(f"Build error: {e}")
            return False, "sbt not found"

    def run_project(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(["sbt", "run"],
                                cwd=self.output_directory,
                                capture_output=True,
                                text=True,
                                check=True)
            logger.info("Successfully ran snippet")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            # Include both stderr and stdout in error output for better debugging
            error_msg = f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            logger.error(f"Run error: {error_msg}")
            return False, error_msg
        except FileNotFoundError:
            error_msg = "sbt not found"
            logger.error(error_msg)
            return False, error_msg

    def evaluate_generated_code(self, dataset_path, run_flag) -> tuple:
        dataset = self.load_json_dataset(dataset_path)
        if dataset is None:
            logger.error("No dataset file provided. Stopping further processing.")
            raise ValueError("No dataset file provided.")

        return self.__calc_working_code_samples(dataset, run_flag)

    def __calc_working_code_samples(self, dataset, run_flag):
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

            with open(self.main_scala_path, "w") as scala_file:
                scala_file.write(code)

            # Run the Scala project
            if run_flag:
                run_status, run_output = self.run_project()
                if not run_status:
                    logger.error("Code causing error!")
                    logger.error(f"Run output: \n{run_output}")
                else:
                    running_examples += 1
        logger.debug(f"Running examples: {running_examples}/{len(results)}")
        return running_examples, len(results)

    def _load_processed_hashes(self):
        if os.path.exists(self.hash_file_path):
            with open(self.hash_file_path) as f:
                return set(json.load(f))
        return set()

    def _save_failing_snippets(self, snippets):
        with open(self.failing_snippets_path, "w") as f:
            json.dump(snippets, f, indent=2)

    def _get_prompt_and_code(self, conversation):
        return (
            [m["value"] for m in conversation.get("conversations",[])
             if m.get("from")=="assistant"],
            [m["value"] for m in conversation.get("conversations",[])
             if m.get("from")=="human"]
        )
