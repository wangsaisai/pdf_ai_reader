# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-PDF Chat AI Agent — a Streamlit web app that lets users upload multiple PDFs and ask questions about their content using RAG (Retrieval-Augmented Generation). Uses Google Gemini as the LLM, LangChain for orchestration, PyPDF2 for text extraction, and FAISS for vector similarity search.

## Running the App

```bash
pip install -r requirements.txt
streamlit run chatapp.py
```

Requires a `GOOGLE_API_KEY` in a `.env` file at the project root (from https://makersuite.google.com/app/apikey).

No tests, linting, or CI/CD are configured.

## Architecture

RAG pipeline with this data flow:

1. **PDF upload** (Streamlit sidebar) → `get_pdf_text()` extracts text via PyPDF2
2. **Chunking** → `get_text_chunks()` uses `RecursiveCharacterTextSplitter` (chunk_size=50000, overlap=1000)
3. **Embedding + indexing** → `get_vector_store()` creates FAISS index using `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`, saves to `faiss_index/`
4. **Question answering** → `user_input()` loads FAISS index, runs `similarity_search()`, then passes results to a LangChain "stuff" QA chain with `ChatGoogleGenerativeAI`

All functions are defined at module level in `chatapp.py` — there are no classes, no separate modules, and no package structure.

## Key Files

- **`chatapp.py`** — Sole application file. Uses `gemini-2.0-flash-exp` model, `langchain_community.vectorstores.FAISS`, `allow_dangerous_deserialization=True` on load, and session state (`st.session_state.processed`) to gate question input until PDFs are processed. Validates `GOOGLE_API_KEY` at startup. Caches embeddings in session state.
- **`faiss_index/`** — Pre-built FAISS vector store (index.faiss + index.pkl). Overwritten when new PDFs are processed.
- **`docs/`** — Sample PDFs for testing.
- **`img/`** — UI images and README screenshots.
