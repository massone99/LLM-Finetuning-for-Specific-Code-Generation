import requests
import json
from typing import Dict, List
import time

class PromptEvaluator:
    def __init__(self, model_name: str = "llama3.2"):
        self.base_url = "http://localhost:11434/api"
        self.model_name = model_name

    def generate_response(self, prompt: str, system_prompt: str = None) -> Dict:
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}

        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(f"{self.base_url}/generate", json=payload)
        return response.json()

    def evaluate_prompt_variations(
        self, base_prompt: str, variations: List[Dict]
    ) -> List[Dict]:
        results = []

        for variant in variations:
            print(f"\nTesting prompt variation: {variant['name']}")

            start_time = time.time()
            response = self.generate_response(
                variant["prompt"], variant.get("system_prompt")
            )
            end_time = time.time()

            results.append(
                {
                    "variation_name": variant["name"],
                    "prompt": variant["prompt"],
                    "system_prompt": variant.get("system_prompt"),
                    "response": response["response"],
                    "time_taken": end_time - start_time,
                }
            )

        return results

