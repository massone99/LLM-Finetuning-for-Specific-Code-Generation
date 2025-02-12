from datasets import load_dataset
from unsloth.chat_templates import (
    standardize_sharegpt,
)

def prepare_dataset(dataset_path, tokenizer):
    """Prepare the dataset for training."""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = standardize_sharegpt(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
