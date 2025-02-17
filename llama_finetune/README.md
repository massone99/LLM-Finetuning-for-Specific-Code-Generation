# Llama Fine-tuning: Code Summary

This directory implements a fine-tuning project for Llama 3.2 3B for DSL tasks.

## Summary of Functions

- Training Pipeline:
  - Loads and prepares datasets.
  - Configures training parameters and manages trainers.
  - Saves fine-tuned models and training metrics.

- Hyperparameter Optimization:
  - Implements grid search using SMAC to tune hyperparameters such as LoRA settings and learning rates. (Still buggy, needs to be fixed)

- Evaluation Utilities:
  - Provides functions to compute evaluation metrics (BLEU, running_snippets/total_snippets) and generate code outputs.
  - Offers tools for analyzing model performance and trends.

- GUI Tools:
  - Includes a PyQt5-based interface for selecting and trimming dataset files.

- Utilities:
  - Contains logging functionality.
  - Offers helper functions for data formatting and conversation extraction.
