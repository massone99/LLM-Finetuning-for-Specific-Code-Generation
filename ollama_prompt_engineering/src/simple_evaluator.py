import requests
import time
from typing import Dict, List
import logging

class SimpleModelEvaluator:
    def __init__(self, model_name: str = "qwen2.5", system_prompt: str = None):
        self.base_url = "http://localhost:11434/api"
        self.model_name = model_name
        self.system_prompt = (
            system_prompt
            or "You are an expert code assistant specializing in Akka and Scala. Your responses must consist solely of syntactically correct, runnable Scala code—no explanations, comments, or markdown formatting are allowed."
        )
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("System prompt: %s", self.system_prompt)

    def generate_response(self, prompt: str) -> Dict:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "system": self.system_prompt
        }

        response = requests.post(f"{self.base_url}/generate", json=payload)
        return response.json()

    def evaluate_prompts(self, prompts: List[str]) -> List[Dict]:
        results = []

        for idx, prompt in enumerate(prompts, 1):
            print(f"\nEvaluating prompt {idx}/{len(prompts)}")
            
            start_time = time.time()
            response = self.generate_response(prompt)
            end_time = time.time()

            results.append({
                "prompt": prompt,
                "system_prompt": self.system_prompt,
                "response": response["response"],
                "time_taken": end_time - start_time,
            })

        return results
