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

def main():
    # Custom CSS to make text areas and layout more readable
    st.markdown("""
    <style>
    .stTextArea textarea {
        height: 300px !important;
        font-size: 14px !important;
    }
    .stTextInput input {
        font-size: 14px !important;
    }
    .stMarkdown {
        font-size: 14px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 Coding Assistant with Pinecone RAG")
    
    # API Key Inputs
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
    
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
        # Validate API keys
        if not (openai_api_key and pinecone_api_key):
            st.error("Please enter all API keys")
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
            rag_assistant = RAGAssistant(openai_api_key, pinecone_api_key)
            
            # Embed documents
            rag_assistant.embed_documents(chunks)
            
            st.success("Documents processed and embedded successfully!")
        
        except Exception as e:
            st.error(f"Error processing documents: {e}")
    
    # Query Section
    st.header("Ask Your Coding Question")
    query = st.text_area("Enter your coding-related query", height=150)
    
    if st.button("Get Answer"):
        # Validate API keys
        if not (openai_api_key and pinecone_api_key):
            st.error("Please enter all API keys")
            return
        
        try:
            # Initialize RAG Assistant
            rag_assistant = RAGAssistant(openai_api_key, pinecone_api_key)
            
            # Retrieve context
            context = rag_assistant.retrieve_context(query)
            
            # Generate response
            response = rag_assistant.generate_response(query, context)
            
            st.subheader("Response")
            st.write(response)
            
            st.subheader("Retrieved Context")
            for i, ctx in enumerate(context, 1):
                st.text_area(f"Context {i}", value=ctx, height=300)
        
        except Exception as e:
            st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
