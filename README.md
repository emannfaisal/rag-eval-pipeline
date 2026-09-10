# Multi-Book RAG Evaluation Pipeline & Chatbot

A production-grade Retrieval-Augmented Generation (RAG) system featuring **Cross-Encoder Reranking**, **Ragas evaluation framework**, and a **modern light-mode single-screen Web UI**.

![RAG System Architecture](https://img.shields.io/badge/RAG-BGE%20%2B%20CrossEncoder-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20gpt--oss--120b-black)
![UI](https://img.shields.io/badge/Frontend-FastAPI%20%2B%20TailwindCSS-emerald)

---

## 🌟 Key Features

* **Multi-Book Context Switching**: Inline dropdown to target queries to specific domain books:
  * 📖 *The Little Book of Deep Learning* (François Fleuret)
  * 🤖 *AI Engineering: Building Applications with Foundation Models* (Chip Huyen)
* **High-Precision Retrieval & Reranking**:
  * Dense vector similarity retrieval via `BAAI/bge-base-en-v1.5` & `ChromaDB`.
  * Re-scoring candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Zero Parametric Memory Leakage**: Strict prompt constraints enforcing zero outside knowledge usage.
* **Persistent Vector Store Reuse**: Automatic caching of embeddings to disk (`./chroma_db`).
* **Automated Benchmarking with Ragas**: Reference-free evaluation scoring **Faithfulness** (0.93), **Answer Relevancy** (0.92), and **Context Precision** (0.71).
* **Modern Monochrome Web UI**: Clean light-mode user interface with warm orange accents, expandable source citations (`[Page X | Heading | Score]`), and auto-scrolling conversation bubbles.

---

## 🛠️ Project Structure

```text
rag-eval-pipeline/
├── rag.py               # Core RAG pipeline, extraction, ChromaDB, Reranker & Ragas eval
├── server.py            # FastAPI web server serving API endpoints & static UI
├── static/
│   ├── index.html       # Single-screen light mode Web UI
│   ├── styles.css       # Custom typography, animations & scrollbar styles
│   └── app.js           # Async chat interactions, book selector & source accordions
├── .env.example         # Template for environment secrets
└── .gitignore           # Git ignore configuration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/emannfaisal/rag-eval-pipeline.git
cd rag-eval-pipeline

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Web Chatbot Interface

Start the FastAPI backend server:

```bash
python server.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 📊 Ragas Evaluation Metrics

| Metric | Score | Impact |
| :--- | :---: | :--- |
| **Faithfulness** | **0.93** | Answers strictly adhere to retrieved PDF context without hallucination. |
| **Answer Relevancy** | **0.92** | High correlation between generated responses and user prompts. |
| **Context Precision** | **0.71** | Cross-encoder reranking filters noisy chunks out of the top-$K$ context window. |

---

## 📜 License
MIT License. Created by Emann Faisal.
