from pathlib import Path
import sys
import json
from typing import List, Dict
# Use relative import for prompt_evaluator
from prompt_evaluator import PromptEvaluator

ROOT_DIR = Path(__file__).parent.parent.parent
LLAMA_FINETUNE_DIR = ROOT_DIR / "llama_finetune" / "llama_finetune"
sys.path.extend([str(ROOT_DIR), str(LLAMA_FINETUNE_DIR)])

from evaluation_utils.build_client import BuildCheckerClient

# Import helper functions
from helpers import process_variation_result, print_prompt_summary, save_evaluation_results, save_final_report

def load_test_prompts(test_set_path: str) -> List[str]:
    with open(test_set_path, 'r') as f:
        test_set = json.load(f)
    return [conv["conversations"][0]["value"] for conv in test_set]



def main():
    evaluator = PromptEvaluator()
    debug_response = False
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
            successful_runs, total_snippet = process_variation_result(
                result, client, variation_metrics, prompt_variation_metrics, debug_response
            )
            prompt_successful_runs += successful_runs
            prompt_total_snippets += total_snippet
            total_successful_runs += successful_runs
            total_snippets += total_snippet

        print_prompt_summary(prompt_index, prompt_successful_runs, prompt_total_snippets, prompt_variation_metrics)

    # Print and save final summary
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
    
    # Save all results
    save_evaluation_results(all_results)
    save_final_report(total_successful_runs, total_snippets, variation_metrics)

if __name__ == "__main__":
    main()
