import json
from pathlib import Path
import sys
from typing import Dict, List

ROOT_DIR = Path(__file__).parent.parent.parent
LLAMA_FINETUNE_DIR = ROOT_DIR / "llama_finetune" / "llama_finetune"
sys.path.extend([str(ROOT_DIR), str(LLAMA_FINETUNE_DIR)])

from evaluation_utils.metrics_calculator import MetricsCalculator

def process_variation_result(result, client, variation_metrics, prompt_variation_metrics, debug_response):
    print("\n" + "-" * 30)
    print(f"Variation: {result['variation_name']}")
    print(f"Time taken: {result['time_taken']:.2f} seconds")
    response = result["response"]
    code = response.strip('`').replace('scala\n', '', 1) if response.startswith('```') else response
    if debug_response:
        print("\nResponse:")
        print(code)
    reference = result["prompt"]
    bleu_score = MetricsCalculator.compute_bleu(reference, code)
    print(f"BLEU Score: {bleu_score:.4f}")
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
    variation_metrics[result['variation_name']]['bleu_scores'].append(bleu_score)
    data_content = json.dumps([{
        "conversations": [
            {"from": "human", "value": result["prompt"]},
            {"from": "assistant", "value": code}
        ]
    }])
    print("\033[92m" + data_content + "\033[0m")
    successful_runs, total_snippet = client.process_dataset_inline_content(
        json.loads(data_content), run=True
    )
    print(f"Build Check: {successful_runs}/{total_snippet} snippets ran successfully")
    variation_metrics[result['variation_name']]['successful_runs'] += successful_runs
    variation_metrics[result['variation_name']]['total_runs'] += total_snippet
    prompt_variation_metrics[result['variation_name']]['successful_runs'] = successful_runs
    prompt_variation_metrics[result['variation_name']]['total_runs'] = total_snippet
    return successful_runs, total_snippet

def print_prompt_summary(prompt_index, prompt_successful_runs, prompt_total_snippets, prompt_variation_metrics):
    print("\n" + "=" * 50)
    print(f"PROMPT {prompt_index} RESULTS")
    print("=" * 50)
    print(f"\nTotal running code snippets for this prompt: {prompt_successful_runs}/{prompt_total_snippets}")
    print("\nMetrics by variation type for this prompt:")
    for variation_name, metrics in prompt_variation_metrics.items():
        success_rate = metrics['successful_runs'] / metrics['total_runs'] * 100 if metrics['total_runs'] else 0
        print(f"\n{variation_name}:")
        print(f"- BLEU Score: {metrics['bleu_score']:.4f}")
        print(f"- Success Rate: {success_rate:.2f}% ({metrics['successful_runs']}/{metrics['total_runs']} snippets)")

def save_evaluation_results(all_results: List[Dict], output_file: Path = None) -> None:
    # New method to save evaluation results to a JSON file.
    if output_file is None:
        output_file = Path(__file__).parent.parent / "res" / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved evaluation results to {output_file}")