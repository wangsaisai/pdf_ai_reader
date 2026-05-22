# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-PDF Chat AI Agent — a Streamlit web app that lets users upload multiple PDFs and ask questions about their content using RAG (Retrieval-Augmented Generation). Supports multiple LLM/embedding providers (Google Gemini, OpenAI, Anthropic, Ollama) via environment variables. Uses LangChain for orchestration, PyPDF2 for text extraction, and FAISS for vector similarity search.

## Running the App

```bash
pip install -r requirements.txt
streamlit run chatapp.py
```

Requires provider API key(s) in a `.env` file at the project root. Set `LLM_PROVIDER` and `EMBEDDING_PROVIDER` to choose between `google`, `openai`, `anthropic`, `ollama`. See `.env.example` for all available environment variables and model examples per provider.

No tests, linting, or CI/CD are configured.

## Architecture

RAG pipeline with this data flow:

1. **PDF upload** (Streamlit sidebar) → `get_pdf_text()` extracts text via PyPDF2
2. **Chunking** → `get_text_chunks()` uses `RecursiveCharacterTextSplitter` (chunk_size=50000, overlap=1000)
3. **Embedding + indexing** → `get_vector_store()` → `create_embeddings()` factory creates provider-specific embeddings, FAISS index saved to `faiss_index/`
4. **Question answering** → `user_input()` loads FAISS index, runs `similarity_search()`, then passes results to a LangChain "stuff" QA chain via `create_llm()` factory

All functions are defined at module level in `chatapp.py` — there are no classes, no separate modules, and no package structure.

## Key Files

- **`chatapp.py`** — Sole application file. Uses `langchain_community.vectorstores.FAISS`, `allow_dangerous_deserialization=True` on load. Session state gates question input until PDFs are processed. Provider-specific LLM/embedding classes loaded via `create_llm()` and `create_embeddings()` factories with lazy imports. Validates API keys at startup based on selected providers.
- **`faiss_index/`** — Pre-built FAISS vector store (index.faiss + index.pkl). Overwritten when new PDFs are processed.
- **`docs/`** — Sample PDFs for testing.
- **`img/`** — UI images and README screenshots.
