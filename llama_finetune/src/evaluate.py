import json
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import codebleu
import tempfile
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

def compute_codebleu(reference, candidate, language="scala"):
    """Compute CodeBLEU score between reference and candidate code."""
    import codebleu
    
    with tempfile.NamedTemporaryFile('w', delete=False, suffix=f'.{language}') as ref_file, \
         tempfile.NamedTemporaryFile('w', delete=False, suffix=f'.{language}') as cand_file:
        ref_file.write(reference)
        cand_file.write(candidate)
        ref_file_path = ref_file.name
        cand_file_path = cand_file.name

    metrics = ["token_overlap", "syntax", "dataflow"]

    try:
        codebleu_score = codebleu.calc_codebleu(
            ref_file_path,
            cand_file_path,
            [language],
            # calc_codebleu() got an unexpected keyword argument 'metrics'
            metrics=metrics
        )
    except Exception as e:
        print(f"Error computing CodeBLEU: {e}")
        codebleu_score = 0.0

    os.remove(ref_file_path)
    os.remove(cand_file_path)
    return codebleu_score

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
    """Evaluate model performance using BLEU and CodeBLEU metrics."""
    from evaluate import compute_bleu, compute_codebleu
    from datasets import load_dataset

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
    codebleu_scores = []
    for ref, gen in zip(test_references, generated_codes):
        bleu = compute_bleu(ref, gen)
        codebleu = compute_codebleu(ref, gen)
        bleu_scores.append(bleu)
        codebleu_scores.append(codebleu)

    # Save results
    # Create results directory if it doesn't exist
    os.makedirs('data/results', exist_ok=True)
    
    results_df = pd.DataFrame({
        'prompt': test_prompts,
        'reference': test_references,
        'generated': generated_codes,
        'BLEU': bleu_scores,
        'CodeBLEU': codebleu_scores
    })
    results_df.to_csv(f'data/results/{output_prefix}_results.csv', index=False)

    # Calculate and save average metrics
    avg_metrics = {
        'average_BLEU': sum(bleu_scores) / len(bleu_scores),
        'average_CodeBLEU': sum(codebleu_scores) / len(codebleu_scores)
    }
    
    with open(f'data/results/{output_prefix}_metrics.json', 'w') as f:
        json.dump(avg_metrics, f, indent=4)

    return avg_metrics

def main():
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_metrics = evaluate_model(
        model, 
        tokenizer, 
        "./data/test_set.json",
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
        "./data/test_set.json",
        "finetuned"
    )
    print("Fine-tuned metrics:", finetuned_metrics)

if __name__ == "__main__":
    main()
