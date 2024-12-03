# Llama Projects

This repository contains a unified collection of Llama-based projects:

1. **Dataset Builder**: Tools for creating and preprocessing training datasets
2. **Llama Fine-tuning**: Scripts and utilities for fine-tuning Llama models
3. **Llama RAG**: Retrieval-Augmented Generation implementation using Llama

## Project Structure

```
.
├── dataset_builder/     # Dataset creation and preprocessing
├── llama_finetune/     # Fine-tuning scripts and configurations
└── llama3_rag_project/ # RAG implementation
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

3. Activate the virtual environment:
```bash
poetry shell
```

## Usage

Each subproject can be used independently within the unified environment:

- **Dataset Builder**: Tools for creating training datasets
- **Llama Fine-tuning**: Fine-tune Llama models on custom datasets
- **RAG**: Implement retrieval-augmented generation using Llama

See individual project directories for specific usage instructions.
