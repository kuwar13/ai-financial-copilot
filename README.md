# AI Financial Operations Copilot

A GenAI + RAG based financial operations assistant for analyzing invoice CSVs, payment documents, and policy files.

## Features

- Upload CSV, PDF, and TXT financial documents
- Ask natural-language questions about invoice/payment CSV data
- Retrieve policy context using RAG
- Validate invoices using an agentic AI workflow
- Preview uploaded CSV records
- Uses Groq LLM, LangChain, ChromaDB, and Hugging Face embeddings

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Groq API
- pandas

## How It Works

1. User uploads financial documents.
2. Documents are chunked and embedded using Hugging Face embeddings.
3. ChromaDB stores document vectors for retrieval.
4. CSV data is passed directly to the LLM for small-table analysis.
5. RAG retrieves relevant policy/document context.
6. Groq LLM generates grounded financial answers and invoice validation recommendations.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
