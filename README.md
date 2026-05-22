# Multi-PDF Chat AI Agent

A Streamlit web app that lets you upload multiple PDFs and ask questions about their content using RAG (Retrieval-Augmented Generation).

Supports multiple LLM/embedding providers: **Google Gemini**, **OpenAI**, **Anthropic**, **Ollama**. Configure via environment variables.

## How It Works

1. **PDF Upload** — Upload one or more PDF files via the sidebar
2. **Text Extraction** — PyPDF2 extracts text page by page, preserving page metadata
3. **Chunking** — Text is split into ~1500-character chunks with 200-character overlap using LangChain's `RecursiveCharacterTextSplitter`
4. **Embedding & Indexing** — Chunks are embedded and stored in a FAISS vector index
5. **Question Answering** — Your question is matched against the index via similarity search; the top 4 relevant chunks (with source and page info) are passed to the LLM to generate an answer

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env    # edit with your API key(s)
streamlit run chatapp.py
```

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in your values.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `google` | LLM provider: `google`, `openai`, `anthropic`, `ollama` |
| `EMBEDDING_PROVIDER` | `google` | Embedding provider: `google`, `openai`, `ollama` |
| `LLM_MODEL` | `gemini-2.0-flash-exp` | Model name for the selected LLM provider |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Model name for the selected embedding provider |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature (0.0 - 1.0) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (only for ollama provider) |
| `OPENAI_BASE_URL` | *(empty)* | Custom base URL for OpenAI-compatible providers |
| `ANTHROPIC_BASE_URL` | *(empty)* | Custom base URL for Anthropic-compatible providers |

### API Keys

Set the key matching your chosen provider:

- `GOOGLE_API_KEY` — for Google Gemini
- `OPENAI_API_KEY` — for OpenAI
- `ANTHROPIC_API_KEY` — for Anthropic
- Ollama requires no API key (runs locally)

See `.env.example` for model name examples per provider.

## Project Structure

```
chatapp.py              # Main application (single file)
requirements.txt        # Python dependencies
.env.example            # Environment variable template
faiss_index/            # FAISS vector store (generated at runtime)
docs/                   # Sample PDFs for testing
```

## Tech Stack

- **Streamlit** — Web UI
- **LangChain** — RAG orchestration, text splitting, QA chain
- **PyPDF2** — PDF text extraction
- **FAISS** — Vector similarity search
- **LLM Providers** — Google Gemini, OpenAI, Anthropic, Ollama

## License

MIT
