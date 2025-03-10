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
        self.failing_snippets_path = (
            build_checker_path / "res/config/failing_snippets.json"
        )
        self.main_scala_path = os.path.join(
            akka_project_path, "src/main/scala/Main.scala"
        )

    def load_json_dataset(self, json_file_path) -> dict:
        if not os.path.exists(json_file_path):
            logger.error(f"The file '{json_file_path}' does not exist.")
            return None
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)
                # If the data contains evaluation results, extract the generated code samples
                if isinstance(data, dict) and "detailed_results" in data:
                    return [
                        {
                            "conversations": [
                                {"from": "assistant", "value": result["generated"]}
                            ]
                        }
                        for result in data["detailed_results"]
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
            is_multi_snippet = len(assistant_msgs) > 1

            if is_multi_snippet:
                logger.info(
                    f"\nProcessing multi-snippet conversation {idx} ({len(assistant_msgs)} snippets)"
                )
            else:
                logger.info(f"\nProcessing single-snippet conversation {idx}")

            for snippet_idx, (code, prompt) in enumerate(
                zip(assistant_msgs, human_prompts)
            ):
                if is_multi_snippet:
                    logger.info(
                        f"Testing snippet {snippet_idx + 1}/{len(assistant_msgs)}"
                    )
                    logger.info(f"Context up to this point:\n{prompt}")

                logger.debug(f"Processing code:\n{code}")

                success, msg = self.test_single_snippet(
                    code,
                    build_flag,
                    run_flag,
                    idx=idx,
                    prompt=prompt,
                    snippet_idx=snippet_idx if is_multi_snippet else None,
                )
                if success:
                    successful_runs += 1
                else:
                    failing_snippets.append(
                        {"idx": idx, "prompt": prompt, "code": code, "error": msg}
                    )
                total_snippets += 1

        self._save_failing_snippets(failing_snippets)
        return successful_runs, total_snippets

    def test_single_snippet(
        self, code: str, build=True, run=True, idx=None, prompt=None, snippet_idx=None
    ) -> tuple[bool, str]:
        if not code.strip():
            return False, "No code provided"

        # Clean up and unwrap the code instead of just removing special tokens
        code = self.clean_and_unwrap_code(code)

        if snippet_idx is not None:
            snippet_info = f"conversation {idx}, snippet {snippet_idx + 1}"
        else:
            snippet_info = f"conversation {idx}"

        logger.info(f"Testing {snippet_info}...")

        # Log only first 100 chars if single snippet, otherwise log the differential prompt
        if snippet_idx is None:
            logger.info(f"Prompt: {prompt[:100]}...")
        else:
            # Get just the last human message from the conversation context
            last_prompt = prompt.split("Human: ")[-1]
            logger.info(f"Current prompt: {last_prompt}")

        with open(self.main_scala_path, "w") as f:
            f.write(code)
            logger.debug(f"Wrote code to {self.main_scala_path}")

        logger.debug(f"Processing code:\n{code}")

        if build:
            build_success, build_msg = self.build_project()
            if not build_success:
                logger.error(f"Build failed for {snippet_info}")
                return False, f"Build failed: {build_msg}"

        if run:
            success, msg, stdout = self.run_project()  # Modified to return stdout
            if not success:
                logger.error(f"Run failed for {snippet_info}")
                logger.error(f"Run output: {msg}")
                # Save failed snippet with its output
                failing_snippet = {
                    "idx": idx,
                    "prompt": prompt,
                    "code": code,
                    "error_output": msg,
                    "run_output": stdout,  # Add the stdout from the failed run
                }
                self._save_failing_snippet(failing_snippet)
            else:
                logger.info(f"Successfully ran {snippet_info}")
            return success, msg

        return True, "Code written successfully"

    def build_project(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["sbt", "compile"],
                cwd=self.output_directory,
                check=True,
                capture_output=True,
                text=True,
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr
            logger.error(f"Build error: {error_msg}")
            return False, error_msg
        except FileNotFoundError as e:
            logger.error(f"Build error: {e}")
            return False, "sbt not found"

    def run_project(self) -> tuple[bool, str, str]:  # Modified return type
        try:
            result = subprocess.run(
                ["sbt", "run"],
                cwd=self.output_directory,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("Successfully ran snippet")
            return True, result.stdout, result.stdout
        except subprocess.CalledProcessError as e:
            # Include both stderr and stdout in error output for better debugging
            error_msg = f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            logger.error(f"Run error: {error_msg}")
            return False, error_msg, e.stdout  # Return stdout separately
        except FileNotFoundError:
            error_msg = "sbt not found"
            logger.error(error_msg)
            return False, error_msg, ""

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
            # Code cleaning is already integrated here, no changes needed
            code = self.clean_and_unwrap_code(code)

            with open(self.main_scala_path, "w") as scala_file:
                scala_file.write(code)

            # Run the Scala project
            if run_flag:
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

    def _save_failing_snippet(self, snippet):
        """Save a single failing snippet by appending it to the failing_snippets.json file"""
        failing_snippets = []
        if os.path.exists(self.failing_snippets_path):
            with open(self.failing_snippets_path, "r") as f:
                try:
                    failing_snippets = json.load(f)
                except json.JSONDecodeError:
                    failing_snippets = []

        failing_snippets.append(snippet)

        with open(self.failing_snippets_path, "w") as f:
            json.dump(failing_snippets, f, indent=2)

    def _get_prompt_and_code(self, conversation):
        """Get all prompts and responses from a conversation, combining previous context"""
        messages = conversation.get("conversations", [])
        prompts = []
        codes = []
        current_prompt = []

        for msg in messages:
            if msg.get("from") == "human":
                # Add human message to current context
                current_prompt.append(f"Human: {msg['value']}")
                prompts.append(
                    "\n".join(current_prompt)
                )  # Always add the current context
            elif msg.get("from") == "assistant":
                codes.append(msg["value"])

        return codes, prompts
    
    def clean_and_unwrap_code(self, code: str) -> str:
        """
        Removes the <|endoftext|> token and unwraps the code from a Scala markdown code block.

        Args:
            code (str): The code snippet to clean.

        Returns:
            str: The unwrapped Scala code, or the original code if no markdown block is found.
        """
        code = code.replace("<|endoftext|>", "").strip()
        # Remove markdown code block delimiters
        if code.startswith("```scala") and code.endswith("```"):
            code = code[8:-3].strip()  # Remove ```scala and ```
        return code
