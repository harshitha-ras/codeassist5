import os
import streamlit as st
import chromadb
import tiktoken
from typing import List
import openai
from openai import OpenAI
import requests
import io
import PyPDF2
import docx
import urllib.parse

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
    def __init__(self, openai_api_key: str):
        """Initialize RAG Assistant with OpenAI API and ChromaDB"""
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB client
        chroma_client = chromadb.Client()
        self.collection = chroma_client.create_collection(name="code_docs")

    def embed_documents(self, documents: List[str]):
        """Embed documents using OpenAI embeddings"""
        for i, doc in enumerate(documents):
            embedding = self.client.embeddings.create(
                input=doc,
                model="text-embedding-ada-002"
            ).data[0].embedding
            
            self.collection.add(
                embeddings=[embedding],
                documents=[doc],
                ids=[f"doc_{i}"]
            )

    def retrieve_context(self, query: str, top_k: int = 3):
        """Retrieve most relevant documents using semantic search"""
        query_embedding = self.client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        ).data[0].embedding
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0]

    def generate_response(self, query: str, context: List[str]):
        """Generate response using OpenAI with retrieved context"""
        context_str = "\n\n".join(context)
        
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Use the provided context to help answer the user's question."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        return response.choices[0].message.content

def main():
    st.title("🚀 Coding Assistant with RAG")
    
    # OpenAI API Key Input
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    # Document Upload Section
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files", 
        type=['pdf', 'docx', 'txt'], 
        accept_multiple_files=True
    )
    
    # URL Input
    url_input = st.text_input("Or enter a URL to extract text")
    
    if st.button("Process Documents"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key")
            return
        
        try:
            documents = []
            
            # Process uploaded files
            for uploaded_file in uploaded_files:
                if uploaded_file.type == 'application/pdf':
                    text = DocumentProcessor.parse_pdf(uploaded_file)
                elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    text = DocumentProcessor.parse_docx(uploaded_file)
                elif uploaded_file.type == 'text/plain':
                    text = DocumentProcessor.parse_txt(uploaded_file)
                documents.append(text)
            
            # Process URL if provided
            if url_input:
                url_text = DocumentProcessor.parse_url(url_input)
                documents.append(url_text)
            
            # Split documents into chunks
            chunks = []
            for doc in documents:
                chunks.extend(TextSplitter.split_text(doc))
            
            # Initialize RAG Assistant
            rag_assistant = RAGAssistant(openai_api_key)
            
            # Embed documents
            rag_assistant.embed_documents(chunks)
            
            st.success("Documents processed and embedded successfully!")
        
        except Exception as e:
            st.error(f"Error processing documents: {e}")
    
    # Query Section
    st.header("Ask Your Coding Question")
    query = st.text_area("Enter your coding-related query")
    
    if st.button("Get Answer"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API Key")
            return
        
        try:
            # Initialize RAG Assistant
            rag_assistant = RAGAssistant(openai_api_key)
            
            # Retrieve context
            context = rag_assistant.retrieve_context(query)
            
            # Generate response
            response = rag_assistant.generate_response(query, context)
            
            st.subheader("Response")
            st.write(response)
            
            st.subheader("Retrieved Context")
            for i, ctx in enumerate(context, 1):
                st.text_area(f"Context {i}", value=ctx, height=100)
        
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
