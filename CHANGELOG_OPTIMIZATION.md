# Optimization History

## Round 1: Code Quality (commit 394c2c9)

**Error handling & validation**
- API key validation at startup with clear error messages
- PDF text extraction: skip pages that return `None`
- Empty upload guard: warn if no PDFs selected before processing
- Empty text guard: error if extracted text is blank
- `user_input()` wrapped in try/except with user-facing error display
- Session state `processed` flag: disable question input until PDFs are processed

**Dependency & import cleanup**
- `langchain` → `langchain_community` (deprecated package migration)
- Dependencies pinned in `requirements.txt`
- Removed stale `pdfreader.py` module

**Embedding reuse**
- `get_embeddings()` caches `GoogleGenerativeAIEmbeddings` in `st.session_state`

---

## Round 2: Multi-Provider Support (commit fb71499)

- Extracted `create_llm()` and `create_embeddings()` factory functions
- Lazy imports per provider (Google, OpenAI, Anthropic, Ollama)
- All config via env vars: `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`, etc.
- Provider-specific API key validation at startup
- Created `.env.example` with full configuration reference

---

## Round 3: RAG Pipeline Optimization (current)

### 3.1 Document Chunking — `chunk_size` 50000 → 1500

**Problem:** `chunk_size=50000` meant most PDFs produced 1-2 chunks total, defeating the purpose of vector search. Retrieval was essentially "return the whole document" — high noise, low precision.

**Fix:**
- `chunk_size`: 50000 → **1500** (roughly 300-400 tokens, a good fit for embedding models)
- `chunk_overlap`: 1000 → **200** (enough context continuity without redundancy)

**Effect:** More chunks → finer-grained retrieval → LLM receives only relevant passages → better answers, lower token cost.

### 3.2 Page Metadata Preservation

**Problem:** All page text was concatenated into a single string, losing page boundaries. No way to tell the user which page an answer came from.

**Fix:**
- `get_pdf_text()` returns a list of `{"text", "source", "page"}` dicts instead of a flat string
- `get_text_chunks()` attaches `metadata` (source filename, page number) to each chunk
- `get_vector_store()` passes `metadatas` to `FAISS.from_texts()`

**Effect:** Each chunk in the vector store carries its origin. Future work can surface page references in answers.

### 3.3 Prompt Template Typo

**Problem:** `{context}` was followed by a stray `?` — `Context:\n {context}?\n` — which subtly confuses the LLM (implies the context is a question).

**Fix:** Removed the `?`.

### 3.4 LLM Instance Caching

**Problem:** `get_conversational_chain()` called `create_llm()` on every question, creating a new LLM client each time.

**Fix:** Added `get_llm()` that caches the LLM instance in `st.session_state.llm`.

### 3.5 Explicit `k` for Similarity Search

**Problem:** `similarity_search()` used the default `k=4` without documentation. With the old 50K chunks, that meant ~200K characters of context.

**Fix:** Explicit `k=4` parameter. Now with 1500-char chunks, that's ~6K characters — well within model context limits.
