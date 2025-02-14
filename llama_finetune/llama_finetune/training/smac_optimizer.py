from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario
from typing import Dict, Any, Callable
import numpy as np

def create_configspace() -> ConfigurationSpace:
    """Create configuration space for SMAC"""
    cs = ConfigurationSpace(seed=42)
    
    # PEFT parameters
    cs.add_hyperparameter(UniformIntegerHyperparameter("r", lower=16, upper=64, default_value=32))
    cs.add_hyperparameter(UniformIntegerHyperparameter("lora_alpha", lower=64, upper=512, default_value=128))
    
    # Training parameters
    cs.add_hyperparameter(UniformIntegerHyperparameter("per_device_train_batch_size", lower=2, upper=4, default_value=4))
    cs.add_hyperparameter(UniformIntegerHyperparameter("num_train_epochs", lower=3, upper=5, default_value=3))
    cs.add_hyperparameter(UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=5e-4, default_value=1e-4, log=True))
    
    return cs

def optimize_hyperparameters(train_function: Callable, n_trials: int = 20, n_workers: int = 1):
    """Run SMAC optimization"""
    configspace = create_configspace()
    
    # Setup SMAC scenario
    scenario = Scenario(
        configspace=configspace,
        deterministic=True,  # Changed to True since we handle seed explicitly
        n_trials=n_trials,
        n_workers=n_workers,
        name="llama_finetune_opt",
        seed=42  # Added explicit seed for reproducibility
    )
    
    # Initialize SMAC optimizer with seed handling
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=train_function,
        overwrite=True
        # Rimosso l'argomento 'backend' non supportato
    )
    
    # Run optimization
    incumbent = smac.optimize()
    
    return incumbent, smac.runhistory

def split_config_dict(config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split configuration into PEFT and training parameters"""
    peft_params = {
        "r": config["r"],
        "lora_alpha": config["lora_alpha"],
    }
    
    training_params = {
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "num_train_epochs": config["num_train_epochs"],
        "learning_rate": config["learning_rate"],
    }
    
    # Add seed if present in config
    if "seed" in config:
        training_params["seed"] = config["seed"]
    
    return peft_params, training_params
