# Llama3 RAG Web Application

This project implements a Retrieval-Augmented Generation (RAG) system using Llama3 model and Gradio interface.

## Setup

1. Install Ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Pull required models:
```bash
ollama serve & ollama pull llama3 & ollama pull nomic-embed-text
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the application:
```bash
python src/main.py
```

2. Open the Gradio interface in your browser (the URL will be displayed in the terminal)

3. Enter your questions in the text box to query the RAG system

## Features

- Web content retrieval and processing
- Document chunking and embedding
- RAG-based question answering
- User-friendly Gradio interface
