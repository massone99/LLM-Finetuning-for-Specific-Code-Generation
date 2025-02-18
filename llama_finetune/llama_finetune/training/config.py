from unsloth import is_bfloat16_supported
from itertools import product

# Grid search configurations - ridotto per testare solo 10 combinazioni
# GRID_SEARCH_PARAMS = {
#     'peft': {
#         'r': [32, 64],  # LoRA rank values - entrambi hanno mostrato buoni risultati
#         'lora_alpha': [128, 512],  # 128: good performance, 512: best performance
#     },
#     'training': {
#         'per_device_train_batch_size': [4],  # 4: good balance
#         'num_train_epochs': [3, 5],  # Test both shorter and longer training
#         'learning_rate': [1e-4, 2e-4],  # Test both conservative and aggressive learning
#     }
# }

# Grid search configurations - optimized for RTX 3070 Laptop GPU
GRID_SEARCH_PARAMS = {
    'peft': {
        'r': [16, 32],  # Reduced LoRA rank values for memory efficiency
        'lora_alpha': [64, 128],  # Lower alpha values for stable training
    },
    'training': {
        'per_device_train_batch_size': [4],  # Keep 4 as it shows good balance
        'num_train_epochs': [3, 4],  # Adjusted based on convergence in logs
        'learning_rate': [5e-5, 1e-4],  # More conservative learning rates
        'max_steps': [-1],  # Add max_steps to grid search params
    }
}

# Base PEFT parameters - updated for stability
BASE_PEFT_PARAMS = {
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",  # Keep for VRAM optimization
    "random_state": 3407,
    "use_rslora": False,
    "loftq_config": None,
}

# Base training parameters - optimized for convergence
BASE_TRAINING_PARAMS = {
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,  # Increased for better initialization
    "max_steps": -1,  # Set to -1 instead of None to use num_train_epochs
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",  # Changed to cosine for better convergence
    "seed": 3407,
    "report_to": "none",
}

def get_grid_combinations():
    """Generate all combinations for grid search"""
    peft_keys = GRID_SEARCH_PARAMS['peft'].keys()
    training_keys = GRID_SEARCH_PARAMS['training'].keys()
    
    peft_values = [GRID_SEARCH_PARAMS['peft'][k] for k in peft_keys]
    training_values = [GRID_SEARCH_PARAMS['training'][k] for k in training_keys]
    
    peft_combinations = list(product(*peft_values))
    training_combinations = list(product(*training_values))
    
    all_combinations = []
    for p_combo in peft_combinations:
        for t_combo in training_combinations:
            peft_dict = dict(zip(peft_keys, p_combo))
            training_dict = dict(zip(training_keys, t_combo))
            all_combinations.append((peft_dict, training_dict))
    
    return all_combinations

def get_peft_params(peft_updates):
    """Merge base PEFT params with updates"""
    params = BASE_PEFT_PARAMS.copy()
    params.update(peft_updates)
    return params

def get_training_params(training_updates):
    """Merge base training params with updates"""
    params = BASE_TRAINING_PARAMS.copy()
    params.update(training_updates)
    return params

# PEFT (Parameter-Efficient Fine-Tuning) parameters - results gathered from experiments
PEFT_PARAMS = {
    "r": 64,  # LoRA rank - 64: Running examples 6/17 - 128: crashes PC
    "target_modules": [  # Layers to apply LoRA
                       "q_proj",
                       "k_proj",
                       "v_proj",
                       "o_proj",  # Attention layers
                       "gate_proj",
                       "up_proj",
                       "down_proj",  # MLP layers
                       # "lm_head"            # Optional: language model head
                       ],
    # lora_alpha results:
    # - 16: basic performance
    # - 128: good performance
    # - 512: best performance, BLEU: 0.36-0.37, Running examples: 8-10/17
    # - 1024: worse performance, Running examples: 7/17
    "lora_alpha": 512,  # α > rank amplifies updates influence
    "lora_dropout": 0,  # Optimized when 0
    "bias": "none",  # Optimized when "none"
    "use_gradient_checkpointing": "unsloth",  # 30% less VRAM usage
    "random_state": 3407,  # Fixed seed for reproducibility
    "use_rslora": False,  # Rank-stabilized LoRA disabled
    "loftq_config": None,  # LoftQ quantization disabled
}

# Training hyperparameters - results gathered from experiments
TRAINING_PARAMS = {
    # Batch sizes tested:
    # - 2: works but slow
    # - 4: good balance
    # - 8: crashes due to memory
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    # Steps configuration:
    # - max_steps=500, warmup_steps=50: mediocre results (BLEU: 0.38, 5/17)
    # - max_steps=60, warmup_steps=5: better results
    "warmup_steps": 5,
    # "num_train_epochs": 4, original with max_steps = 60
    "num_train_epochs": 15, # TODO: provare a ottenere una double descent overfittando su un numero bello alto di epoche: superiamo le 5 epoche, penso che 10 potrebbero sorprendermi
    # "max_steps": 60, # TODO: PROVA -1 PER USARE IL NUMERO DI EPOCHE
    "max_steps": -1,
    "learning_rate": 2e-4,
    "fp16": not is_bfloat16_supported(),  # Use FP16 if bfloat16 not available
    "bf16": is_bfloat16_supported(),  # Prefer bfloat16 if supported
    "logging_steps": 1,
    "optim": "adamw_8bit",  # 8-bit AdamW for memory efficiency
    "weight_decay": 0.01,  # L2 regularization
    "lr_scheduler_type": "linear",  # Linear learning rate decay
    "seed": 3407,  # Fixed seed for reproducibility
    "report_to": "none",  # Disable external logging
}

def _store_model_record(output_dir, model_hash, record):
    """
    Internal utility to store a model record in the trained_models.json file.
    """
    trained_models_path = os.path.join(output_dir, "trained_models.json")

    if os.path.exists(trained_models_path):
        with open(trained_models_path, "r") as f:
            trained_models = json.load(f)
    else:
        trained_models = []
        os.makedirs(os.path.dirname(trained_models_path), exist_ok=True)

    if any(m["model_hash"] == model_hash for m in trained_models):
        print(f"Model info already exists for hash: {model_hash}")
    else:
        trained_models.append(record)
        with open(trained_models_path, "w") as f:
            json.dump(trained_models, f, indent=2)
        print(f"Stored new model info with hash: {model_hash}")

def store_model_info(output_dir, train_dataset_size, test_dataset_size, trainer_args, avg_bleu, samples_info, peft_params=PEFT_PARAMS):
    """
    Compute a unique hash for the model and hyperparameters, and append
    model info to 'trained_models.json' if this hash isn't present yet.
    """
    import os
    import hashlib
    from datetime import datetime 
    import json

    info_str = (
        f"batch_size={trainer_args.per_device_train_batch_size}-"
        f"lr={trainer_args.learning_rate}-"
        f"epochs={trainer_args.num_train_epochs}-"
        f"max_steps={trainer_args.max_steps}-"
        f"train_size={train_dataset_size}-"
        f"test_size={test_dataset_size}-"
        f"lora_rank={peft_params['r']}-"
        f"lora_alpha={peft_params['lora_alpha']}-"
        f"lora_dropout={peft_params['lora_dropout']}"
    )
    model_hash = hashlib.sha256(info_str.encode("utf-8")).hexdigest()

    record = {
        "model_hash": model_hash,
        "timestamp": datetime.now().isoformat(),
        "model_type": "finetuned",
        # Training parameters
        "batch_size": trainer_args.per_device_train_batch_size,
        "gradient_accumulation_steps": trainer_args.gradient_accumulation_steps,
        "learning_rate": trainer_args.learning_rate,
        "num_train_epochs": trainer_args.num_train_epochs,
        "max_steps": trainer_args.max_steps,
        "warmup_steps": trainer_args.warmup_steps,
        "optimizer": trainer_args.optim,
        "weight_decay": trainer_args.weight_decay,
        "lr_scheduler": trainer_args.lr_scheduler_type,
        # PEFT parameters
        "lora_rank": peft_params["r"],
        "lora_alpha": peft_params["lora_alpha"],
        "lora_dropout": peft_params["lora_dropout"],
        "target_modules": peft_params["target_modules"],
        "use_gradient_checkpointing": peft_params["use_gradient_checkpointing"],
        # Dataset info
        "train_dataset_size": train_dataset_size,
        "test_dataset_size": test_dataset_size,
        # Evaluation metrics
        "avg_bleu": avg_bleu,
        "execution_check": {
            "successful_runs": samples_info[0],
            "total_snippets": samples_info[1]
        },  
    }
    
    _store_model_record(output_dir, model_hash, record)

def store_base_model_info(output_dir, train_dataset_size, test_dataset_size, avg_bleu, samples_info):
    """
    Store model info for the base (not fine-tuned) model.
    """
    import os
    import hashlib
    from datetime import datetime 
    import json
    
    info_str = f"base_model-train_size={train_dataset_size}-test_size={test_dataset_size}"
    model_hash = hashlib.sha256(info_str.encode("utf-8")).hexdigest()

    record = {
        "model_hash": model_hash,
        "timestamp": datetime.now().isoformat(),
        "model_type": "base",
        # Dataset info
        "train_dataset_size": train_dataset_size,
        "test_dataset_size": test_dataset_size,
        # Evaluation metrics
        "avg_bleu": avg_bleu,
        "execution_check": {
            "successful_runs": samples_info[0],
            "total_snippets": samples_info[1]
        },
    }

    _store_model_record(output_dir, model_hash, record)

