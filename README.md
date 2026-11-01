# Nexus Workplace Services — Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) system built to query and extract answers from Nexus Workplace Services internal policy documents with precision, source citations, and zero-downtime resilience.

Powered by **ChromaDB** for vector storage, **Streamlit** for the web frontend, and an intelligent **Dual-Engine LLM & Embedding Pipeline** that uses **OpenAI** as the primary engine and automatically falls back to **Google Gemini** if the OpenAI API key or quota fails.

---

## 📋 Table of Contents
1. [Architecture & Workflow](#architecture--workflow)
2. [Document Collection & Chunking Strategy](#document-collection--chunking-strategy)
3. [Dual Vector DB & Fallback Strategy](#dual-vector-db--fallback-strategy)
4. [Project Structure](#project-structure)
5. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
6. [Running the Application](#running-the-application)
7. [Running Tests](#running-tests)
8. [Design Decisions & Justifications](#design-decisions--justifications)

---

## 🏛️ Architecture & Workflow

```
[ User Query via Streamlit ]
              │
              ▼
   ┌──────────────────────┐
   │ Query Embedding      │
   │ (OpenAI or Gemini)   │
   └──────────┬───────────┘
              │ Vector (1536d / 768d)
              ▼
   ┌──────────────────────────────────────────────┐
   │ ChromaDB Persistent Vector Store             │
   │ ├── nexus_policies_openai  (OpenAI Vectors)  │
   │ └── nexus_policies_gemini  (Gemini Vectors)  │
   └──────────────────┬───────────────────────────┘
                      │ Top-K Most Relevant Sections
                      ▼
   ┌──────────────────────────────────────────────┐
   │ Prompt Construction                          │
   │ (Enriched Policy Context + Citations Prompt) │
   └──────────────────┬───────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────┐
        │ Primary: OpenAI (gpt-4o-mini)│
        └──────────────┬──────────────┘
                       │ (On error / quota exceeded / timeout)
                       ▼
        ┌─────────────────────────────┐
        │ Fallback: Gemini 1.5 Flash  │
        └──────────────┬──────────────┘
                       │
                       ▼
[ Streamlit UI with Citations & Fallback Status Badge ]
```

---

## 📄 Document Collection & Chunking Strategy

### 1. The Knowledge Base
The system indexes three official policy documents from the `knowledge-base/` directory:
- **`Expense_Reimbursement_Guide.txt`** (`FIN-EXP-006` | Version 1.4 | Effective 1 Oct 2026)
- **`IT_Acceptable_Use_Policy.txt`** (`IT-SEC-002` | Version 3.0 | Effective 15 Aug 2026)
- **`Staff_Travel_Policy.txt`** (`HR-TRV-004` | Version 2.1 | Effective 1 Sep 2026)

### 2. Chunking Strategy: Section-Based Semantic Chunking with Context Enrichment
Rather than using arbitrary sliding token/character windows (e.g. 500 characters with 50-character overlap), we implemented a **structural section-based semantic chunker**:

- **Why Not Fixed-Character Chunking?** Fixed-size chunking frequently cuts through sentences or splices clauses across different policy rules. In policy documents, an employee needs the full rule (e.g., receipt thresholds and requirements) intact.
- **Why Not Full-Document Chunking?** Entire documents dilute vector precision and force the LLM to search through extensive noise, increasing latency and hallucination risk.
- **Section Parsing**: Corporate policies are divided into clearly numbered, self-contained clauses (`1. ELIGIBLE EXPENSES`, `2. NON-ELIGIBLE EXPENSES`, etc.). We parse these headers using regular expressions.
- **Context Enrichment**: To ensure the vector representation retains full provenance, each chunk's text is prefixed with document-level metadata:
  ```text
  Policy: Staff Travel Policy (Doc ID: HR-TRV-004)
  Effective: 1 September 2026 | Version: 2.1

  4. CLAIMS
  Expense claims must be submitted within 10 working days after the trip ends.
  Receipts are required for individual expenses above ₦10,000.
  ...
  ```
- **Metadata Stored in ChromaDB**: Each chunk in ChromaDB stores:
  - `source_file`: File name
  - `doc_title`: Formal document title
  - `doc_id`: Official ID (e.g., `FIN-EXP-006`)
  - `version`: Policy version
  - `effective_date`: Policy effective date
  - `section_number`: Numerical section index
  - `section_title`: Full section header

---

## 🔄 Dual Vector DB & Fallback Strategy

A key engineering challenge when implementing a fallback from OpenAI to Gemini is **vector space incompatibility**:
- OpenAI `text-embedding-3-small` generates **1536-dimensional** vectors.
- Google Gemini `text-embedding-004` generates **768-dimensional** vectors.

Comparing an OpenAI query vector against Gemini document vectors (or vice versa) is mathematically invalid and rejected by vector databases.

### The Solution: Dual-Index Architecture
1. **Parallel ChromaDB Collections**:
   - `nexus_policies_openai`: Indexed with OpenAI embeddings.
   - `nexus_policies_gemini`: Indexed with Gemini embeddings.
2. **Transparent Fallback at Runtime**:
   - The query attempts OpenAI retrieval first.
   - If OpenAI encounters an error (rate limits, quota exhaustion, network error, or invalid key), the system instantly catches the exception and routes the query to Gemini embeddings + the Gemini Chroma collection.
   - Generation similarly attempts OpenAI `gpt-4o-mini` first and smoothly switches to Gemini `gemini-1.5-flash` if any error occurs.
   - The UI displays an explicit status badge indicating whether the answer was generated by the primary model or the fallback model.

---

## 📁 Project Structure

```
teaching-RAG/
├── knowledge-base/                    # Source corporate policy documents
│   ├── Expense_Reimbursement_Guide.txt
│   ├── IT_Acceptable_Use_Policy.txt
│   └── Staff_Travel_Policy.txt
├── chroma_db/                         # Persistent ChromaDB storage (generated)
├── venv/                              # Python virtual environment
├── .env                               # API keys and model configurations
├── .gitignore                         # Git ignore configuration
├── requirements.txt                   # Project dependencies
├── config.py                          # Global configuration and environment loader
├── chunking.py                        # Document loader and semantic section chunker
├── embeddings.py                      # OpenAI & Gemini embedding providers
├── vector_store.py                    # ChromaDB collection manager and retrieval logic
├── llm_client.py                      # Primary & fallback LLM generation handlers
├── rag_engine.py                      # High-level RAG orchestrator
├── app.py                             # Streamlit web application frontend
└── test_rag.py                        # Verification and testing script
```

---

## 🛠️ Prerequisites & Environment Setup

### 1. Python Environment
Python 3.10+ is recommended. Ensure your virtual environment is active:

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables (`.env`)
Create or verify your `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
```

> **Note**: Both keys are supported. If the OpenAI key expires or is empty, Gemini takes over automatically.

---

## 🚀 Running the Application

Launch the Streamlit frontend with:

```bash
.\venv\Scripts\streamlit run app.py
```

### Features of the Frontend:
- **Interactive Chat Interface**: Ask questions in natural language.
- **Sample Queries**: One-click prompt suggestions to immediately test common policy questions.
- **Execution Mode Toggle**:
  - `Auto`: Normal primary & fallback operation.
  - `Force OpenAI`: Directly query OpenAI.
  - `Force Gemini`: Test the fallback pipeline on demand.
- **Retrieval Inspector**: Expandable drawer showing the retrieved policy clauses, similarity scores, and Document IDs.
- **Real-Time Model Badges**: Indicates whether OpenAI or the Gemini fallback handled the query.
- **Re-index Button**: Re-builds the ChromaDB vector database directly from the UI.

---

## 🧪 Running Tests

Run the test suite to verify document parsing, ChromaDB indexing, retrieval accuracy, and both LLM providers:

```bash
.\venv\Scripts\python test_rag.py
```

---

## 💡 Design Decisions & Justifications

| Feature / Decision | Choice Made | Why? |
| :--- | :--- | :--- |
| **Vector Store** | **ChromaDB (Persistent)** | Lightweight, serverless, native Python support, fast HNSW cosine search, zero external infrastructure required. |
| **Chunking** | **Section-Based Semantic Chunking** | Policy documents have logical section boundaries. This preserves complete clauses and eliminates fragmented sentences. |
| **Context Enrichment** | **Header Metadata Prepending** | Prepending policy title and Doc ID into chunk text gives embeddings domain context, avoiding cross-policy confusion. |
| **Fallback Handling** | **Dual ChromaDB Collections** | Resolves the vector dimension mismatch between OpenAI (1536d) and Gemini (768d) with zero retrieval latency. |
| **LLM Grounding** | **Strict System Prompt & Citations** | Instructs the model to answer strictly from context and cite Document ID and Section number to prevent hallucinations. |
| **Frontend** | **Streamlit** | Rapid, reactive UI with built-in chat components, sidebar controls, and state management without frontend boilerplate. |

