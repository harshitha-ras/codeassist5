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

# [Keep all existing classes: DocumentProcessor, TextSplitter, RAGAssistant]

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
