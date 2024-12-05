from datetime import datetime
import json
import os

import gradio as gr
import nltk
import numpy as np
import ollama
import pandas as pd
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# TODO: duplicated code from evaluate.py
def compute_bleu(reference, candidate):
    """Compute BLEU score between reference and candidate code."""
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

# TODO: duplicated code from evaluate.py
def evaluate_rag_system(rag_system, test_dataset_path):
        """Evaluate RAG system performance using BLEU metric."""
        # Load and prepare test dataset
        test_dataset = load_dataset('json', data_files=test_dataset_path, split='train')
        test_df = test_dataset.to_pandas()

        # Extract prompts and references
        def extract_conversations(convo_list):
            try:
                human_msg = convo_list[0]['value']
                assistant_msg = convo_list[1]['value']
                return pd.Series({'prompt': human_msg, 'reference': assistant_msg})
            except (IndexError, KeyError, TypeError) as e:
                print(f"Error extracting conversations: {e}")
                return pd.Series({'prompt': None, 'reference': None})

        test_df[['prompt', 'reference']] = test_df['conversations'].apply(extract_conversations)
        test_df.dropna(subset=['prompt', 'reference'], inplace=True)

        # Generate answers using RAG
        test_prompts = test_df['prompt'].tolist()
        test_references = test_df['reference'].tolist()
        generated_answers = [rag_system.get_answer(prompt) for prompt in test_prompts]

        # Calculate BLEU scores
        bleu_scores = []
        results = []

        for ref, gen in zip(test_references, generated_answers):
            bleu = compute_bleu(ref, gen)
            bleu_scores.append(bleu)

            results.append({
                'reference': ref,
                'generated': gen,
                'bleu': bleu
            })

        # Calculate average scores
        avg_bleu = sum(bleu_scores) / len(bleu_scores)

        # Save results
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'average_metrics': {'bleu': avg_bleu},
            'detailed_results': results
        }

        os.makedirs('./data/rag_results', exist_ok=True)
        output_file = f'./data/rag_results/rag_evaluation_{datetime.now().strftime("%Y%m%d")}.json'
        
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"\nRAG Evaluation Results:")
        print(f"Average BLEU Score: {avg_bleu:.4f}")
        print(f"Detailed results saved to: {output_file}")

        return evaluation_results

class RAGSystem:
    def __init__(self, urls):
        self.urls = urls if isinstance(urls, list) else [urls]
        self.setup_rag()

    def setup_rag(self):
        # Load and process documents from all URLs
        all_docs = []
        for url in self.urls:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        # Create embeddings and vector store
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def calculate_relevancy_score(self, question: str, context: str) -> float:
        """Calculate relevancy score between question and retrieved context using embedding similarity"""
        question_embedding = self.embeddings.embed_query(question)
        context_embedding = self.embeddings.embed_query(context)
        
        # Calculate cosine similarity
        similarity = np.dot(question_embedding, context_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(context_embedding)
        )
        return float(similarity)
    
    def ollama_llm(self, question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[{'role': 'user', 'content': formatted_prompt}]
        )
        return response['message']['content']

    def get_answer(self, question):
        retrieved_docs = self.retriever.invoke(question)
        formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return self.ollama_llm(question, formatted_context)

def main():
    # Initialize RAG system with multiple URLs
    urls = [
        'https://doc.akka.io/libraries/akka-core/current/typed/actors.html',
        'https://doc.akka.io/libraries/akka-core/current/typed/actor-lifecycle.html',
        'https://doc.akka.io/libraries/akka-core/current/typed/interaction-patterns.html'
    ]
    rag_system = RAGSystem(urls)

    # Evaluate RAG system
    evaluation_results = evaluate_rag_system(
        rag_system,
        test_dataset_path="./data/test_set.json"
    )
    interface = False
    
    if interface:
        # Create Gradio interface
        iface = gr.Interface(
            fn=rag_system.get_answer,
            inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
            outputs="text",
            title="RAG with Llama3",
            description="Ask questions about the provided context",
        )

        # Launch the interface
        iface.launch()

if __name__ == "__main__":
    main()
