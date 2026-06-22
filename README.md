# 🤖 RAG Chatbot using Gemini, FAISS & Streamlit

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, Google Gemini, FAISS, and HuggingFace Embeddings.

The chatbot allows users to:

* Upload PDF documents
* Load website content using URLs
* Create a vector database using FAISS
* Retrieve relevant information from uploaded content
* Generate context-aware responses using Gemini

---

## Features

* 📄 Multiple PDF Upload Support
* 🌐 Website URL Content Loading
* 🔍 Semantic Search using FAISS
* 🧠 HuggingFace Embeddings
* 🤖 Gemini 2.5 Flash Integration
* 💬 Interactive Chat Interface
* ⚡ Streamlit Frontend
* 📚 Retrieval-Augmented Generation (RAG)

---

## Tech Stack

### Frontend

* Streamlit

### LLM

* Google Gemini 2.5 Flash

### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

### Vector Database

* FAISS

### Document Loaders

* PyPDFLoader
* WebBaseLoader

### Programming Language

* Python

---

## Project Architecture

User Query
↓
Retriever
↓
FAISS Vector Database
↓
Relevant Chunks
↓
Gemini 2.5 Flash
↓
Response

---

## Installation

### Clone Repository

git clone https://github.com/Nvijayvamsi/rag-chatbot-gemini.git

cd rag-chatbot-gemini

### Create Virtual Environment

python -m venv .venv

### Activate Environment

Windows

.venv\Scripts\activate

### Install Dependencies

pip install -r requirements.txt

---

## Environment Variables

Create a `.env` file in the project root.

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

---

## Run Application

streamlit run app.py

---

## Usage

1. Upload one or more PDF files.
2. Enter a website URL (optional).
3. Click Process.
4. Wait for vector database creation.
5. Ask questions related to the uploaded documents or website.
6. Receive AI-generated responses grounded in the provided content.

---

## Example Questions

* What is Retrieval-Augmented Generation?
* Summarize the uploaded document.
* What are the key features mentioned in the website?
* Explain the main concepts discussed in the PDF.

---

## Future Enhancements

* Chat Memory
* Source Citations
* Multiple Website Crawling
* Hybrid Search (BM25 + Vector Search)
* User Authentication
* Database Integration
* Cloud Deployment

---


