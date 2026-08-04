# Union Bank of India - Enterprise Banking RAG System

A production-grade Banking Retrieval-Augmented Generation (RAG) system built with **Clean Architecture**, **SOLID Principles**, and domain-specific regulatory capabilities.

---

## 📁 Recommended Datasets & Knowledge Sources

The system is optimized for Indian Banking Regulatory & Operational documents across 5 key domains:

| Domain | Key Regulators & Authorities | Recommended Datasets & Documents to Use |
|---|---|---|
| **AML / KYC** | Reserve Bank of India (RBI), FIU-IND, PMLA (MHA) | • RBI Master Direction - Know Your Customer (KYC) Direction, 2016<br>• Prevention of Money-Laundering Act (PMLA), 2002<br>• FIU-IND Guidance Notes on STR / CTR Reporting |
| **Compliance** | RBI, SEBI, Ministry of Corporate Affairs (MCA) | • RBI Master Directions on Financial Inclusion, Credit Cards, Microfinance<br>• SEBI (LODR) Regulations, 2015<br>• Companies Act, 2013 |
| **Risk** | RBI, Basel Committee | • RBI Master Circular on Basel III Capital Regulations<br>• Internal Capital Adequacy Assessment Process (ICAAP) Guidelines<br>• Operational Risk & Liquidity Risk Frameworks |
| **Internal Audit** | Union Bank Internal Governance | • Concurrent & Statutory Audit Manuals<br>• Long Form Audit Report (LFAR) Guidelines<br>• Internal Audit Observation Reports |
| **Board Secretariat** | MCA, Union Bank Board | • Board & Committee Meeting Minutes / Agendas<br>• Fair Practices Code & Board Resolutions |

### Where to Place Your PDF/DOCX/TXT Documents
Create a `./data` directory in the project root and place your regulatory PDFs, Word documents, or text files inside:
```bash
mkdir -p data/rbi_directions data/aml_kyc data/internal_policies
```

---

## ⚡ Setup & Prerequisites

### 1. Python Environment Setup
Ensure you have Python 3.10+ installed.

```bash
cd C:\Users\swara\.gemini\antigravity\scratch\banking_rag

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### 2. Install Required Dependencies
```bash
pip install pypdf python-docx sentence-transformers qdrant-client transformers torch pydantic-settings pytest
```

---

## 🚀 How to Run the System

### Step 1: Run Offline Document Ingestion Pipeline
Ingest raw banking documents (PDF, DOCX, TXT). This parses the documents, applies contextual chunking, extracts regulatory metadata, generates BGE-M3 embeddings, and indexes them into Qdrant.

```bash
# Option A: Ingest a whole folder of regulatory documents
python main.py ingest --path ./data

# Option B: Ingest a single PDF file (e.g., RBI Master Direction on KYC)
python main.py ingest --path ./data/rbi_directions/RBI_Master_Direction_KYC.pdf
```

---

### Step 2: Run Online RAG Query Engine

#### Option A: Interactive CLI Assistant Mode
Launch the interactive terminal session to chat with the Union Bank AI Compliance Officer:
```bash
python main.py
```
*Example session output:*
```text
=======================================================
 UNION BANK OF INDIA - BANKING RAG ASSISTANT 
 Serving: Compliance, Risk, Audit, AML/KYC, Board Sec 
=======================================================

Enter Compliance / Banking Question: What are the Customer Due Diligence requirements for high-risk accounts?

-------------------------------------------------------
ANSWER:
1. Regulated Entities (REs) including Union Bank of India must apply Enhanced Due Diligence (EDD) for high-risk customers, including senior management approval and verification of source of funds. [Source: RBI_Master_Direction_KYC.pdf | Regulator: Reserve Bank of India (RBI) | Section: Section 15 | Page: 12]
-------------------------------------------------------
```

#### Option B: Single Command-Line Query
```bash
python main.py query -q "What is the timeline for submitting Suspicious Transaction Reports to FIU-IND?"
```

---

### Step 3: Run Unit & Integration Tests
To verify all chunkers, retrievers, rerankers, filters, and pipelines:
```bash
pytest tests/
```

---

## ⚙️ Environment Variables (Optional Configuration)

You can customize models and Qdrant database settings by setting environment variables or creating a `.env` file:

```env
# Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=union_bank_knowledge_base

# Models
EMBEDDING_MODEL_NAME=BAAI/bge-m3
SLM_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
RERANKER_MODEL_NAME=BAAI/bge-reranker-large
MODEL_DEVICE=cpu # Change to 'cuda' if GPU is available

# Retrieval Defaults
TOP_K_DENSE=20
TOP_K_RERANKED=5
```
