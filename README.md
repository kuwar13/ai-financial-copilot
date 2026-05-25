# 💼 AI Financial Operations Copilot

A GenAI + RAG powered financial operations assistant that analyzes invoice/payment CSV data and company policy TXT files using natural language.

This project demonstrates practical AI engineering for financial workflows using:

- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Groq LLM
- Streamlit
- pandas

The application allows users to upload invoice/payment datasets and custom company policy rules, then interact with them through natural-language queries and AI-powered invoice validation.

---

#  Features

##  Financial CSV Analysis

Upload invoice/payment CSV files and ask questions like:

- Which invoices are unpaid?
- List all high-risk invoices
- Which invoices have missing PO numbers?
- Compare BILL-1001 and BILL-1013
- Which department has the most payment risk?
- Summarize financial risks

---

##  Custom Policy Support

Upload TXT files containing company financial/compliance policies.

Example:

```text
Invoices above $15,000 require director approval.

Missing PO numbers or payment mismatches require manual audit review.

High-risk invoices cannot be auto-approved.

Security and Finance department vendors require additional verification.
```

Then ask:

- Which invoices violate company policy?
- Which invoices require director approval?
- Which invoices should be escalated?
- Which invoices cannot be auto-approved?

---

##  AI Invoice Validation Agent

Paste invoice text and let the AI agent:

- validate invoice compliance
- detect payment mismatches
- identify missing PO numbers
- analyze invoice risk
- apply company policy rules
- generate recommendations

Example output:

```text
Risk Level:
Issues Found:
Recommended Action:
Explanation:
```

---

# GenAI + RAG Architecture

The application combines:
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- LLM Reasoning

Workflow:

```text
CSV/TXT Upload
        ↓
LangChain Loaders
        ↓
Chunking
        ↓
Hugging Face Embeddings
        ↓
ChromaDB Vector Store
        ↓
Semantic Retrieval (RAG)
        ↓
Groq LLM
        ↓
Financial Analysis & Invoice Validation
```

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend |
| Streamlit | Web UI |
| LangChain | RAG orchestration |
| ChromaDB | Vector database |
| Hugging Face Sentence Transformers | Embeddings |
| Groq API | LLM inference |
| pandas | CSV processing |

---




# ⚙️ Setup

## 1. Clone Repository

```bash
git clone https://github.com/kuwar13/ai-financial-copilot.git
cd ai-financial-copilot
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from:

```text
https://console.groq.com/keys
```

---

# Run Application

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

---

#  Usage

## Step 1 — Upload Files

Upload:
- CSV invoice/payment data
- TXT company policy file

---

## Step 2 — Build RAG Index

Click:

```text
Build / Refresh RAG Index
```

This:
- chunks uploaded files
- creates embeddings
- stores vectors in ChromaDB

---

## Step 3 — Select Files

Choose:
- CSV file for invoice/payment analysis
- TXT policy file for company rules

---

## Step 4 — Ask Questions

Example questions:

```text
Summarize policy.
```

```text
Which invoices violate company policy?
```

```text
Which invoices require director approval?
```

```text
Which invoices have missing PO numbers?
```

```text
List all high-risk invoices.
```

```text
Compare BILL-1001 with BILL-1013.
```

```text
Summarize payment risks.
```

---

#  Invoice Validation Example

Paste invoice text:

```text
Invoice Number: BILL-1014
Vendor: CyberShield LLC
PO Number: Missing
Invoice Amount: 21500
Payment Status: Unpaid
Department: Security
Risk Category: High
```

The AI agent will:
- retrieve relevant invoice/payment records
- apply company policy rules
- analyze compliance risk
- generate recommendations

---


#  Why This Project Matters

This project demonstrates:

- Applied GenAI Engineering
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- LLM-powered Financial Analysis
- AI Workflow Automation
- Invoice Validation Agents
- Policy-based Reasoning
- Real-world Financial Operations AI

---

# Author

## Kuwarpreet Singh

- GitHub: https://github.com/kuwar13
- LinkedIn: https://www.linkedin.com/in/kuwarpreet-singh-247827220

---
