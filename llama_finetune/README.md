# Llama 3.2 Fine-tuning Project

This project implements fine-tuning of the Llama 3.2 model for code generation tasks, specifically focused on Scala/Akka code generation.

## Project Structure

```
llama_finetune/
├── data/               # Dataset files
├── src/               # Source code
│   ├── train.py       # Training script
│   └── evaluate.py    # Evaluation script
├── tests/             # Test files
├── outputs/           # Training outputs and model checkpoints
├── pyproject.toml     # Poetry dependencies
└── README.md          # This file
```

## Setup

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Install codebleu separately (due to packaging constraints):
```bash
pip install codebleu==0.1.7
```

4. Place your dataset files:
- Put your training dataset in `data/dataset_llama.json`
- Put your test dataset in `data/test_set.json`

## Usage

### Training

To fine-tune the model:

```bash
poetry run python src/train.py
```

### Evaluation

To evaluate the model:

```bash
poetry run python src/evaluate.py
```

## Model Details

- Base Model: Llama 3.2 3B Instruct
- Fine-tuning Method: LoRA with Unsloth optimizations
- Training Parameters:
  - Learning Rate: 2e-4
  - Batch Size: 2
  - Gradient Accumulation Steps: 4
  - Training Steps: 60

## Evaluation Metrics

The model is evaluated using:
- BLEU Score
- CodeBLEU Score (specifically for code generation)

Results are saved in CSV and JSON format in the project directory.
