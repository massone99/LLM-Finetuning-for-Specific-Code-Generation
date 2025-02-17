# Llama Prompt Engineering

## Overview
This project evaluates various prompt engineering strategies for generating Akka/Scala code using a Llama 3.2 model.

## Components
- **prompt_evaluator.py:** Contains the PromptEvaluator class for generating responses and evaluating prompt variations.
- **main.py:** Executes prompt evaluations, gathers metrics, and checks code snippets using a build checker client.
- **helpers.py:** Processes evaluation results and prints summaries.


## How It Works
1. The evaluator generates multiple variations for each base prompt (e.g., Zero-shot, Chain of Thought, Few-Shot, Tree of Thoughts, and ReAct).
2. Each variation is sent to the Llama3.2-3B model through the Ollama API.
3. Responses are evaluated using BLEU scores and run through a build checker.
4. Metrics are collected and summarized for analysis.

## Usage
- Configure Ollama for serving requests locally.
- Add test prompts in `res/test_set.json`.
- Run `main.py` to evaluate prompts and generate metrics.
