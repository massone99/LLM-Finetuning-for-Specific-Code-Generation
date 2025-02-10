from pathlib import Path
import sys
import requests
import json
from typing import Dict, List
import time
from prompt_evaluator import PromptEvaluator

ROOT_DIR = Path(__file__).parent.parent.parent
LLAMA_FINETUNE_DIR = ROOT_DIR / "llama_finetune" / "llama_finetune"
sys.path.extend([str(ROOT_DIR), str(LLAMA_FINETUNE_DIR)])

from evaluation_utils.metrics_calculator import MetricsCalculator
from evaluation_utils.build_client import BuildCheckerClient
from evaluation_utils.data_processor import DataProcessor

def load_test_prompts(test_set_path: str) -> List[str]:
    with open(test_set_path, 'r') as f:
        test_set = json.load(f)
    return [conv["conversations"][0]["value"] for conv in test_set]

def main():
    evaluator = PromptEvaluator()
    
    debug_response = True
    
    # Load prompts from test_set.json
    test_set_path = Path(__file__).parent.parent / "res" / "test_set.json"
    test_prompts = load_test_prompts(test_set_path)
    
    all_results = []
    total_successful_runs = 0
    total_snippets = 0
    variation_metrics = {}  # Dictionary to store metrics for each variation type
    
    for prompt_index, prompt in enumerate(test_prompts, 1):
        print(f"\nEvaluating prompt {prompt_index}/{len(test_prompts)}: {prompt}")
        print("=" * 50)
        
        # Reset metrics for this prompt
        prompt_variation_metrics = {}
        prompt_successful_runs = 0
        prompt_total_snippets = 0
        
        variations = evaluator.create_variations(prompt)
        results = evaluator.evaluate_prompt_variations(prompt, variations)
        all_results.extend(results)

        client = BuildCheckerClient()  # Create the client for this prompt

        for result in results:
            print("\n" + "-" * 30)
            print(f"Variation: {result['variation_name']}")
            print(f"Time taken: {result['time_taken']:.2f} seconds")
            
            # Get the code response and clean it
            response = result["response"]
            code = response.strip('`').replace('scala\n', '', 1) if response.startswith('```') else response

            if debug_response:        
                print("\nResponse:")
                print(code)

            # Evaluate BLEU score
            reference = result["prompt"]
            bleu_score = MetricsCalculator.compute_bleu(reference, code)
            print(f"BLEU Score: {bleu_score:.4f}")

            # Initialize metrics for this variation
            if result['variation_name'] not in variation_metrics:
                variation_metrics[result['variation_name']] = {
                    'bleu_scores': [],
                    'successful_runs': 0,
                    'total_runs': 0
                }
            if result['variation_name'] not in prompt_variation_metrics:
                prompt_variation_metrics[result['variation_name']] = {
                    'bleu_score': bleu_score,
                    'successful_runs': 0,
                    'total_runs': 0
                }

            # Add metrics
            variation_metrics[result['variation_name']]['bleu_scores'].append(bleu_score)

            # Process the code
            data_content = json.dumps([{
                "conversations": [
                    {"from": "human", "value": result["prompt"]},
                    {"from": "assistant", "value": code}
                ]
            }])
            
            # Print data_content in green
            print("\033[92m" + data_content + "\033[0m")
            

            successful_runs, total_snippet = client.process_dataset_inline_content(
                json.loads(data_content), run=True
            )
            print(f"Build Check: {successful_runs}/{total_snippet} snippets ran successfully")
            
            # Update metrics
            prompt_successful_runs += successful_runs
            prompt_total_snippets += total_snippet
            total_successful_runs += successful_runs
            total_snippets += total_snippet
            
            variation_metrics[result['variation_name']]['successful_runs'] += successful_runs
            variation_metrics[result['variation_name']]['total_runs'] += total_snippet
            prompt_variation_metrics[result['variation_name']]['successful_runs'] = successful_runs
            prompt_variation_metrics[result['variation_name']]['total_runs'] = total_snippet

        # Print prompt summary
        print("\n" + "=" * 50)
        print(f"PROMPT {prompt_index} RESULTS")
        print("=" * 50)
        print(f"\nTotal running code snippets for this prompt: {prompt_successful_runs}/{prompt_total_snippets}")
        
        print("\nMetrics by variation type for this prompt:")
        for variation_name, metrics in prompt_variation_metrics.items():
            success_rate = metrics['successful_runs'] / metrics['total_runs'] * 100 if metrics['total_runs'] > 0 else 0
            print(f"\n{variation_name}:")
            print(f"- BLEU Score: {metrics['bleu_score']:.4f}")
            print(f"- Success Rate: {success_rate:.2f}% ({metrics['successful_runs']}/{metrics['total_runs']} snippets)")

    # Print final summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    print(f"\nTotal running code snippets: {total_successful_runs}/{total_snippets}")
    
    print("\nOverall metrics by variation type:")
    for variation_name, metrics in variation_metrics.items():
        avg_bleu = sum(metrics['bleu_scores']) / len(metrics['bleu_scores'])
        success_rate = metrics['successful_runs'] / metrics['total_runs'] * 100
        print(f"\n{variation_name}:")
        print(f"- Average BLEU Score: {avg_bleu:.4f}")
        print(f"- Success Rate: {success_rate:.2f}% ({metrics['successful_runs']}/{metrics['total_runs']} snippets)")

if __name__ == "__main__":
    main()
