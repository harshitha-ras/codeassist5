import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from typing import List
import openai
from openai import OpenAI
import requests
import io
import PyPDF2
import docx
import urllib.parse
import tiktoken
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge import Rouge

class DocumentProcessor:
    @staticmethod
    def parse_pdf(file):
        """Extract text from PDF files"""
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def parse_docx(file):
        """Extract text from DOCX files"""
        doc = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    @staticmethod
    def parse_txt(file):
        """Extract text from TXT files"""
        return file.read().decode('utf-8')

    @staticmethod
    def parse_url(url):
        """Extract text from web pages"""
        try:
            response = requests.get(url)
            return response.text
        except Exception as e:
            st.error(f"Error fetching URL: {e}")
            return ""

class TextSplitter:
    @staticmethod
    def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(encoding.decode(chunk))
        
        return chunks

class RAGAssistant:
    def __init__(self, openai_api_key: str, pinecone_api_key: str):
        """Initialize RAG Assistant with OpenAI API and Pinecone"""
        # OpenAI Client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Pinecone Initialization
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Create or connect to index
        index_name = "coding-assistant-index"
        
        # Check if index exists, create if not
        if index_name not in [index.name for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name, 
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='gcp',  # Use GCP for free plan
                    region='us-central1'
                )
            )
        
        # Connect to the index
        self.index = self.pc.Index(index_name)

    def embed_documents(self, documents: List[str]):
        """Embed documents using OpenAI embeddings and store in Pinecone"""
        batch_size = 100
        
        for i in range(0, len(documents), batch_size):
            # Select batch of documents
            batch_docs = documents[i:i+batch_size]
            
            # Create embeddings
            embeddings = self.openai_client.embeddings.create(
                input=batch_docs,
                model="text-embedding-ada-002"
            )
            
            # Prepare vectors for Pinecone
            vectors = [
                {
                    'id': f'doc_{i+j}', 
                    'values': embeddings.data[j].embedding,
                    'metadata': {'text': batch_docs[j]}
                } 
                for j in range(len(batch_docs))
            ]
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)

    def retrieve_context(self, query: str, top_k: int = 5):
        """Retrieve most relevant documents using semantic search"""
        # Create query embedding
        query_embedding = self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        ).data[0].embedding
        
        # Perform semantic search
        results = self.index.query(
            vector=query_embedding, 
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract text from metadata
        return [
            match['metadata']['text'] 
            for match in results['matches']
        ]

    def generate_response(self, query: str, context: List[str]):
        """Generate response using OpenAI with retrieved context"""
        context_str = "\n\n".join(context)
        
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Use the provided context to help answer the user's question."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        return response.choices[0].message.content


class RAGEvaluator:
    def __init__(self, rag_assistant):
        self.rag_assistant = rag_assistant
        self.rouge = Rouge()

    def evaluate_retrieval(self, queries, ground_truth):
        precisions, recalls, f1_scores, mrr_scores = [], [], [], []
        total_retrieval_time = 0

        for query, truth in zip(queries, ground_truth):
            start_time = time.time()
            retrieved_context = self.rag_assistant.retrieve_context(query)
            total_retrieval_time += time.time() - start_time

            relevant = [1 if any(t in c for t in truth) else 0 for c in retrieved_context]
            precisions.append(precision_score([1]*len(truth), relevant, zero_division=1))
            recalls.append(recall_score([1]*len(truth), relevant, zero_division=1))
            f1_scores.append(f1_score([1]*len(truth), relevant, zero_division=1))

            for i, context in enumerate(retrieved_context):
                if any(t in context for t in truth):
                    mrr_scores.append(1 / (i + 1))
                    break
            else:
                mrr_scores.append(0)

        return {
            "avg_precision": sum(precisions) / len(precisions),
            "avg_recall": sum(recalls) / len(recalls),
            "avg_f1": sum(f1_scores) / len(f1_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            "avg_retrieval_time": total_retrieval_time / len(queries)
        }

    def evaluate_generation(self, queries, ground_truth):
        rouge_scores, generation_times = [], []

        for query, truth in zip(queries, ground_truth):
            start_time = time.time()
            context = self.rag_assistant.retrieve_context(query)
            generated_response = self.rag_assistant.generate_response(query, context)
            generation_times.append(time.time() - start_time)

            rouge_scores.append(self.rouge.get_scores(generated_response, truth)[0])

        avg_rouge = {
            "rouge-1": sum(score["rouge-1"]["f"] for score in rouge_scores) / len(rouge_scores),
            "rouge-2": sum(score["rouge-2"]["f"] for score in rouge_scores) / len(rouge_scores),
            "rouge-l": sum(score["rouge-l"]["f"] for score in rouge_scores) / len(rouge_scores),
        }

        return {
            "avg_rouge": avg_rouge,
            "avg_generation_time": sum(generation_times) / len(generation_times)
        }

def main():
    # [Keep existing CSS and title]

    st.title("🚀 Coding Assistant with Pinecone RAG")
    
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
    
    # [Keep existing document upload and URL input sections]

    if st.button("Process Documents"):
        # [Keep existing document processing logic]

    st.header("Ask Your Coding Question")
    query = st.text_area("Enter your coding-related query", height=150)
    
    if st.button("Get Answer"):
        if not (openai_api_key and pinecone_api_key):
            st.error("Please enter all API keys")
            return
        
        try:
            rag_assistant = RAGAssistant(openai_api_key, pinecone_api_key)
            context = rag_assistant.retrieve_context(query)
            response = rag_assistant.generate_response(query, context)
            
            st.subheader("Response")
            st.write(response)
            
            st.subheader("Retrieved Context")
            for i, ctx in enumerate(context, 1):
                st.text_area(f"Context {i}", value=ctx, height=300)

            # Evaluation
            evaluator = RAGEvaluator(rag_assistant)
            
            # Sample evaluation data (you should replace this with actual evaluation data)
            evaluation_queries = ["What is a list comprehension in Python?", "Explain decorators in Python"]
            ground_truth = [
                ["List comprehension is a concise way to create lists", "It consists of brackets containing an expression followed by a for clause"],
                ["Decorators are a way to modify or enhance functions", "They use the @decorator syntax in Python"]
            ]

            retrieval_metrics = evaluator.evaluate_retrieval(evaluation_queries, ground_truth)
            generation_metrics = evaluator.evaluate_generation(evaluation_queries, ground_truth)

            st.subheader("Performance Metrics")
            st.json(retrieval_metrics)
            st.json(generation_metrics)
        
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
