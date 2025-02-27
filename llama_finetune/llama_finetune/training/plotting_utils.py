import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Union

def plot_training_metrics(
    metrics: Dict[str, List[float]], 
    output_path: str, 
    train_dataset_size: int,
    peft_params: Dict[str, Union[int, float, str]],
    smooth_factor: int = 5
) -> str:
    """
    Create and save plots showing training convergence.
    
    Args:
        metrics: Dictionary containing lists of metrics (loss, learning_rate, etc.)
        output_path: Directory where to save the plots
        train_dataset_size: Size of the training dataset
        peft_params: PEFT parameters used for training
        smooth_factor: Window size for smoothing the loss curve
    
    Returns:
        Path to the saved plot file
    """
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    if 'loss' in metrics and len(metrics['loss']) > 0:
        plt.subplot(2, 1, 1)
        
        # Get raw loss values
        steps = list(range(1, len(metrics['loss']) + 1))
        loss_values = metrics['loss']
        
        # Plot raw loss as light line
        plt.plot(steps, loss_values, 'lightblue', alpha=0.3, label='Raw loss')
        
        # Calculate and plot smoothed loss
        if len(loss_values) > smooth_factor:
            smoothed_loss = []
            for i in range(len(loss_values)):
                if i < smooth_factor:
                    # For initial points, average what's available
                    smoothed_loss.append(np.mean(loss_values[:i+1]))
                else:
                    # Moving average with window size = smooth_factor
                    smoothed_loss.append(np.mean(loss_values[i-smooth_factor+1:i+1]))
            plt.plot(steps, smoothed_loss, 'b-', label=f'Smoothed loss (window={smooth_factor})')
        
        plt.xlabel('Training steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate
    if 'learning_rate' in metrics and len(metrics['learning_rate']) > 0:
        plt.subplot(2, 1, 2)
        steps = list(range(1, len(metrics['learning_rate']) + 1))
        plt.plot(steps, metrics['learning_rate'], 'g-')
        plt.xlabel('Training steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add information about training parameters
    plt.figtext(
        0.5, 0.01, 
        f"Training dataset size: {train_dataset_size} | " + 
        f"LoRA rank: {peft_params.get('r', 'N/A')} | " + 
        f"LoRA alpha: {peft_params.get('lora_alpha', 'N/A')}",
        ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create filename and save plot
    os.makedirs(output_path, exist_ok=True)
    plot_filename = os.path.join(
        output_path, 
        f"training_convergence_size{train_dataset_size}_r{peft_params.get('r', 'NA')}_alpha{peft_params.get('lora_alpha', 'NA')}.png"
    )
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def plot_comparison(
    model_dirs: List[str],
    base_dir: str,
    output_path: str
) -> Optional[str]:
    """
    Create a comparison plot of multiple models' convergence.
    
    Args:
        model_dirs: List of model directories to compare
        base_dir: Base directory where model outputs are stored
        output_path: Where to save the comparison plot
    
    Returns:
        Path to the saved comparison plot, or None if failed
    """
    plt.figure(figsize=(14, 10))
    
    model_data = []
    for model_dir in model_dirs:
        metrics_path = os.path.join(base_dir, model_dir, "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                data = json.load(f)
                if "metrics_history" in data:
                    model_data.append((model_dir, data["metrics_history"]))
    
    if not model_data:
        return None
    
    # Plot loss comparison
    plt.subplot(2, 1, 1)
    for model_name, metrics in model_data:
        if "loss" in metrics:
            steps = list(range(1, len(metrics["loss"]) + 1))
            plt.plot(steps, metrics["loss"], label=model_name)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save comparison plot
    os.makedirs(output_path, exist_ok=True)
    plot_filename = os.path.join(output_path, "model_convergence_comparison.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename
