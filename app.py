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
    def __init__(self):
        self.text_splitter = TextSplitter()

    def process_file(self, file):
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.pdf':
            return self.process_pdf(file)
        elif file_extension == '.docx':
            return self.process_docx(file)
        elif file_extension == '.txt':
            return self.process_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def process_pdf(self, file):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return self.text_splitter.split_text(text)

    def process_docx(self, file):
        doc = docx.Document(io.BytesIO(file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        return self.text_splitter.split_text(text)

    def process_txt(self, file):
        text = file.read().decode("utf-8")
        return self.text_splitter.split_text(text)

    def process_url(self, url):
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        return self.text_splitter.split_text(text)

class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk))
        return chunks

class RAGAssistant:
    def __init__(self, openai_api_key, pinecone_api_key):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        self.index_name = "coding-assistant-index"
        self.initialize_pinecone()

    def initialize_pinecone(self):
        if self.index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name,
                metric="cosine",
                dimension=1536,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )
        self.index = self.pinecone_client.Index(self.index_name)

    def embed_documents(self, documents: List[str]):
        embeddings = []
        for i in range(0, len(documents), 100):
            batch_docs = documents[i:i+100]
            batch_embeddings = self.openai_client.embeddings.create(
                input=batch_docs,
                model="text-embedding-ada-002"
            )
            embeddings.extend([e.embedding for e in batch_embeddings.data])
        return embeddings

    def index_documents(self, documents: List[str]):
        embeddings = self.embed_documents(documents)
        vectors = [
            (f"doc_{i}", embedding, {"text": doc})
            for i, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]
        self.index.upsert(vectors=vectors)

    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        query_embedding = self.openai_client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        ).data[0].embedding

        results = self.index.query(vector=query_embedding, top_k=k, include_metadata=True)
        return [match.metadata["text"] for match in results.matches]

    def generate_response(self, query: str, context: List[str]) -> str:
        context_str = "\n\n".join(context)
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Use the provided context to help answer the user's question."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
        ]
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500
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
    st.set_page_config(page_title="Coding Assistant with Pinecone RAG", page_icon="🚀", layout="wide")

    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 Coding Assistant with Pinecone RAG")
    
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
    
    st.header("Document Processing")
    
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    url = st.text_input("Or enter a URL")
    
    if st.button("Process Documents"):
        if not (openai_api_key and pinecone_api_key):
            st.error("Please enter all API keys")
            return
        
        try:
            processor = DocumentProcessor()
            rag_assistant = RAGAssistant(openai_api_key, pinecone_api_key)
            
            if uploaded_file:
                chunks = processor.process_file(uploaded_file)
            elif url:
                chunks = processor.process_url(url)
            else:
                st.error("Please upload a file or enter a URL")
                return
            
            rag_assistant.index_documents(chunks)
            st.success("Documents processed and indexed successfully!")
        
        except Exception as e:
            st.error(f"Error processing documents: {e}")

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
