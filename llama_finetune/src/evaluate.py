import json
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

nltk.download('punkt')
nltk.download('punkt_tab')

def compute_bleu(reference, candidate):
    """Compute BLEU score between reference and candidate code."""
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

def generate_code(model, tokenizer, prompts, max_new_tokens=256, temperature=1.5, min_p=0.1):
    """Generate code outputs for a list of prompts."""
    FastLanguageModel.for_inference(model)
    
    generated_codes = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=min_p
        )
        generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generated_codes.append(generated_code)
    return generated_codes

def evaluate_model(model, tokenizer, test_dataset_path, output_prefix="baseline"):
    """Evaluate model performance using BLEU metrics."""
    from evaluate import compute_bleu
    from datasets import load_dataset
    import json
    from datetime import datetime

    # Load test dataset
    test_dataset = load_dataset('json', data_files=test_dataset_path, split='train')
    test_df = test_dataset.to_pandas()

    # Extract prompts and references
    def extract_conversations(convo_list):
        try:
            human_msg = convo_list[0]['value']
            assistant_msg = convo_list[1]['value']
            return pd.Series({'prompt': human_msg, 'reference': assistant_msg})
        except (IndexError, KeyError, TypeError) as e:
            print(f"Error extracting conversations: {e}")
            return pd.Series({'prompt': None, 'reference': None})

    test_df[['prompt', 'reference']] = test_df['conversations'].apply(extract_conversations)
    test_df.dropna(subset=['prompt', 'reference'], inplace=True)

    # Generate code
    test_prompts = test_df['prompt'].tolist()
    test_references = test_df['reference'].tolist()
    generated_codes = generate_code(model, tokenizer, test_prompts)

    # Calculate metrics
    bleu_scores = []
    results = []

    for ref, gen in zip(test_references, generated_codes):
        bleu = compute_bleu(ref, gen)
        bleu_scores.append(bleu)
        
        results.append({
            'reference': ref,
            'generated': gen,
            'bleu': bleu,
        })

    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_metrics = {'bleu': avg_bleu}

    # Prepare evaluation results
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'average_metrics': avg_metrics,
        'detailed_results': results
    }

    # Create results directory if it doesn't exist
    os.makedirs('./data/results', exist_ok=True)

    # Save results
    output_file = f'./data/results/evaluation_results_{output_prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    print(f"\nEvaluation Results ({output_prefix}):")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Detailed results saved to: {output_file}")

    return evaluation_results

def main():
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    test_set_path = "../data/test_set.json"

    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_metrics = evaluate_model(
        model, 
        tokenizer, 
        test_set_path,
        "baseline"
    )
    print("Baseline metrics:", baseline_metrics)
    
    # Load fine-tuned model weights here if evaluating after fine-tuning
    # model.load_state_dict(torch.load("path_to_finetuned_weights"))
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    finetuned_metrics = evaluate_model(
        model, 
        tokenizer, 
        test_set_path,
        "finetuned"
    )
    print("Fine-tuned metrics:", finetuned_metrics)

if __name__ == "__main__":
    main()
