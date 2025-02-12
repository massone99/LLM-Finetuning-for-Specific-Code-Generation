import os
import time
import json
from datasets import load_dataset
from evaluate import evaluate_model
from logger import file_logger
from .dataset_utils import prepare_dataset
from .trainer_setup import setup_trainer
from .config import PEFT_PARAMS, store_model_info

def process_trained_model(
    args,
    max_seq_length,
    model,
    output_dir,
    tokenizer,
    train_dataset_size,
    peft_params=None,
    training_params=None,
):
    # Prepare dataset
    dataset = prepare_dataset(args.dataset_path, tokenizer)

    # Setup and start training with specific parameters
    trainer = setup_trainer(
        model,
        tokenizer,
        dataset,
        max_seq_length,
        output_dir,
        peft_params,
        training_params,
    )

    # Track training time
    start_time = time.time()
    training_output = trainer.train()
    training_time = time.time() - start_time

    # Save training metrics
    training_metrics = {
        "training_time_seconds": training_time,
        "training_stats": training_output.metrics if training_output else None,
    }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    # Save the fine-tuned model
    trainer.save_model(f"{output_dir}/finetuned_model")

    # Evaluate model after fine-tuning
    print("Evaluating model after fine-tuning...")
    output_prefix = f"after_finetuning_trainsize{train_dataset_size}"

    results_file, avg_bleu, samples_info = evaluate_model(
        model, tokenizer, args.test_dataset_path, train_dataset_size, output_prefix
    )

    train_time_str = f"\nTraining completed in {training_time:.2f} seconds"
    train_metrics_path_str = (
        f"\nTraining metrics saved to: {output_dir}/training_metrics.json"
    )

    # Store model info
    store_model_info(
        "../res/data/trained_models/",
        train_dataset_size,
        len(load_dataset("json", data_files=args.test_dataset_path, split="train")),
        trainer.args,
        avg_bleu,
        samples_info,
        PEFT_PARAMS,
    )

    file_logger.write_and_print(train_time_str)
    file_logger.write_and_print(train_metrics_path_str)
