# SLM Finetuning for Library-Specific Code Generation

**Author:** Lorenzo Massone
**Academic Advisor**: Prof. Mirko Viroli
**Co-Advisor**: Prof. Gianluca Aguzzi
**University of Bologna – Cesena Campus**

## Overview

This project demonstrates how to fine-tune **Small Language Models (SLMs)** for generating library-specific code—in particular, Akka/Scala code. The approach allows you to locally adapt general-purpose language models to generate code that is executable and tailored for specific frameworks, without requiring huge computational resources.

## Main Features

* **Parameter-Efficient Fine-Tuning (PEFT):** Uses techniques like LoRA to specialize only a small subset of model parameters.
* **Akka/Scala Focus:** Trains models to generate code patterns for the Akka actor framework.
* **Lightweight Setup:** Experiments show models can be fine-tuned on local hardware (e.g., laptops or Colab).
* **Data Pipeline:** Includes scripts for dataset preparation, fine-tuning, and automatic code validation.

## Models Used

* Llama3.2 3B
* Qwen 2.5 7B

## Results

- Fine-tuned Llama3.2 3B: 70% executable code (vs 0% for base)
- Fine-tuned Qwen 2.5 7B: 88% executable code
