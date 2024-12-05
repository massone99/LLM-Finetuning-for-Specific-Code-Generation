import gradio as gr
import ollama
import time
from bs4 import BeautifulSoup as bs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from typing import Dict, Any
import numpy as np

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
