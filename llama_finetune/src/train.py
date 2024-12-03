import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

def load_model_and_tokenizer(model_name="unsloth/Llama-3.2-3B-Instruct", max_seq_length=2048):
    """Load the model and tokenizer."""
    dtype = None  # Auto detection
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    return model, tokenizer

def prepare_dataset(dataset_path, tokenizer):
    """Prepare the dataset for training."""
    dataset = load_dataset('json', data_files=dataset_path, split='train')
    dataset = standardize_sharegpt(dataset)
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir="outputs"):
    """Set up the SFT trainer."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    
    return trainer

def main():
    # Configuration
    max_seq_length = 2048
    dataset_path = "./data/dataset_llama.json"
    output_dir = "./outputs"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(max_seq_length=max_seq_length)
    
    # Prepare dataset
    dataset = prepare_dataset(dataset_path, tokenizer)
    
    # Setup and start training
    trainer = setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir)
    trainer_stats = trainer.train()
    
    print("Training completed!")
    print(f"Training stats: {trainer_stats}")

if __name__ == "__main__":
    main()
