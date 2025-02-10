import requests
import json
from typing import Dict, List
import time

class PromptEvaluator:
    def __init__(self, model_name: str = "llama3.2"):
        self.base_url = "http://localhost:11434/api"
        self.model_name = model_name
        
    def generate_response(self, prompt: str, system_prompt: str = None) -> Dict:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        response = requests.post(f"{self.base_url}/generate", json=payload)
        return response.json()

    def evaluate_prompt_variations(self, base_prompt: str, variations: List[Dict]) -> List[Dict]:
        results = []
        
        for variant in variations:
            print(f"\nTesting prompt variation: {variant['name']}")
            
            start_time = time.time()
            response = self.generate_response(
                variant['prompt'],
                variant.get('system_prompt')
            )
            end_time = time.time()
            
            results.append({
                'variation_name': variant['name'],
                'prompt': variant['prompt'],
                'system_prompt': variant.get('system_prompt'),
                'response': response['response'],
                'time_taken': end_time - start_time
            })
            
        return results

def main():
    evaluator = PromptEvaluator()
    
    # Example: Testing different prompt engineering techniques
    base_prompt = "Write a function to calculate fibonacci numbers"
    base_system_prompt = "You are a code assistant. Provide only code in your response, without any explanations or markdown formatting. Use Scala programming language."
    
    variations = [
        {
            'name': 'Zero shot',
            'prompt': base_prompt,
            'system_prompt': base_system_prompt
        },
        {
            'name': 'Chain of Thought',
            'prompt': f"Let's solve this step by step:\n1. First, understand what we need:\n{base_prompt}\n2. What are the key components needed?\n3. How should we implement it?",
            'system_prompt': base_system_prompt
        },
        {
            'name': 'Role-based',
            'prompt': base_prompt,
            'system_prompt': f"{base_system_prompt}\nYou are an expert Scala programmer. Provide clean, efficient, and well-documented solutions."
        }
    ]
    
    results = evaluator.evaluate_prompt_variations(base_prompt, variations)
    
    # Print results in a formatted way
    for result in results:
        print("\n" + "="*50)
        print(f"Variation: {result['variation_name']}")
        print(f"Time taken: {result['time_taken']:.2f} seconds")
        print("\nResponse:")
        print(result['response'])

if __name__ == "__main__":
    main()