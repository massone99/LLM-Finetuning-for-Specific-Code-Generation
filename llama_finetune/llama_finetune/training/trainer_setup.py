from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from typing import Dict, List

class MetricsCallback(TrainerCallback):
    """Callback to collect metrics during training"""
    def __init__(self):
        super().__init__()
        self.metrics_history = {
            "loss": [],
            "learning_rate": []
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Save metrics on each logging step"""
        if logs is not None:
            # Extract loss and learning rate
            if "loss" in logs:
                self.metrics_history["loss"].append(logs["loss"])
            if "learning_rate" in logs:
                self.metrics_history["learning_rate"].append(logs["learning_rate"])

def setup_trainer(model, tokenizer, dataset, max_seq_length, output_dir="outputs", peft_params=None, training_params=None):
    """Set up the SFT trainer with specific parameters for grid search"""
    # Use the provided PEFT parameters
    model = FastLanguageModel.get_peft_model(model, **peft_params)

    # Ensure output_dir is set in training parameters
    if training_params is None:
        raise ValueError("Training parameters must be provided")
    training_params["output_dir"] = output_dir

    # Create training arguments
    training_args = TrainingArguments(**training_params)

    # Create metrics callback to track convergence
    metrics_callback = MetricsCallback()

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
        callbacks=[metrics_callback],  # Add our custom callback
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    return trainer, metrics_callback  # Return the callback along with trainer
