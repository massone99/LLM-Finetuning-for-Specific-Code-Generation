from termcolor import colored
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def load_model_and_tokenizer(model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=2048):
    """Load the model and tokenizer using the llama chat template"""

    print(colored(f"Finetuning model: {model_name}", "green"))

    dtype = None  # Auto detection
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Ensure tokenizer has pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
