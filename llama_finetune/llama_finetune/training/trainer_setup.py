from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

def setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir="outputs", peft_params=None, training_params=None):
    """Set up the SFT trainer with specific parameters for grid search"""
    # Use the provided PEFT parameters
    model = FastLanguageModel.get_peft_model(model, **peft_params)

    # Create training arguments
    training_args = TrainingArguments(**training_params)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    return trainer
