

# TICKCHAT

An end-to-end, production-lean RAG (Retrieval-Augmented Generation) assistant for classifying support tickets and answering “product knowledge” questions using a searchable knowledge base built from public documentation.

* **Routable topics** (e.g., *How-to, Product, Best practices, API/SDK, SSO*) → go through the RAG pipeline and return an answer with citations.
* **Everything else** → displays a clear, routed message (e.g., “This ticket has been classified as a ‘Connector’ issue and routed to the appropriate team.”)

---

## Table of contents

1. [Why this project?](#why-this-project)
2. [Architecture](#architecture)
3. [Design decisions & trade-offs](#design-decisions--trade-offs)
4. [Repository layout](#repository-layout)
5. [Prerequisites](#prerequisites)
6. [Quick start (local)](#quick-start-local)
7. [Configuration](#configuration)
8. [Index build: crawl & ingest](#index-build-crawl--ingest)
9. [Run the app](#run-the-app)
10. [How it works (runtime flow)](#how-it-works-runtime-flow)
11. [Troubleshooting](#troubleshooting)
12. [Extending](#extending)
13. [Security notes](#security-notes)
14. [License](#license)

---

## Why this project?

Support bots are often either:

* **LLM-only** (fluent but can hallucinate), or
* **Search-only** (accurate but verbose, not conversational).

This repo combines **document retrieval** with **generative answers** so users get concise, cited, up-to-date responses for the topics that matter, and clean routing for everything else.

---

## Architecture

```
+------------------+        classify         +-------------------+
|  User Question   |  -------------------->  |  Topic Classifier |
+------------------+                         +-------------------+
           |                                        | yes (RAG topics)
           | no (non-RAG topics)                    v
           v                                +------------------+    top-k
  Routed message "X"                        |  Retriever       | <---------+
 (e.g., "Connector")                        |  (Vector DB)     |           |
                                            +------------------+           |
                                                     | chunks              |
                                                     v                     |
                                            +------------------+           |
                                            |  Augment Prompt  |           |
                                            +------------------+           |
                                                     | context             |
                                                     v                     |
                                            +------------------+           |
                                            |   Generator      | ----------+
                                            | (LLM)            |
                                            +------------------+
                                                     |
                                                     v
                                            Answer + citations
```

**Key components**

* `crawler_to_pinecone.py`: crawls allowed sites (e.g., docs pages), cleans & chunks text, embeds, and **upserts into Pinecone**.
* `rag_pipeline.py`: wraps retrieval + prompt assembly + generation + citation formatting.
* `llm_utils.py`: model clients, retry/timeouts, token accounting.
* `models.py`: data classes/configs for requests, responses, and typed settings.
* `app.py`: minimal UI/API that classifies topics and either:

  * calls RAG (for *How-to, Product, Best practices, API/SDK, SSO*), or
  * returns a routed classification message for everything else.

*(File names based on repository listing.)* ([GitHub][1])

---
Perfect 👍 thanks for pasting the actual **RAG pipeline** and **classification pipeline** code.
Now we can be very explicit in the README about exactly which **models** and **vector DB** your repo is wired to.

Here’s the **revised “Design decisions & trade-offs”** section with the models + DBs from your code:

---

## Design decisions & trade-offs

### 1) **Vector database**

* **Used:** [Pinecone](https://www.pinecone.io/), connected to the `atlan-docs` index.
* **Why:**

  * **Pros:** Managed, scalable, serverless; works well on Streamlit Cloud / Replit where persistent local storage isn’t available.
  * **Cons:** Requires API key, introduces external dependency and cost.
* **Alternatives considered:**

  * **FAISS:** Fast, open source, local-first. Great for small datasets and offline mode, but not practical for cloud-hosted apps without storage.
  * **Redis:** In-memory + persistence option, but more ops overhead.
* **Rationale:** Pinecone was chosen to simplify deployment and ensure the vector store is shared across environments.

---

### 2) **Embedding model**

* **Used:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

  * Dimension: **384** (code checks index vs. model dimension at startup).
* **Why:**

  * Light, fast, widely used for semantic search.
  * Open source, so no external API cost for embedding.
* **Alternatives considered:**

  * **OpenAI `text-embedding-3-small` / `-3-large`:** Higher recall but adds API dependency and cost.
* **Rationale:** `all-MiniLM-L6-v2` balances **speed, accuracy, and zero API cost**, which is great for rapid iteration.

---

### 3) **Generator (answering) model**

* **Used:** Groq-hosted **`llama-3.1-8b-instant`** for answer generation in the RAG pipeline.
* **Why:**

  * Low latency, inexpensive inference, good balance of reasoning and throughput.
* **Alternatives considered:**

  * Larger LLaMA models (`70B`) for higher reasoning power but slower/more expensive.
  * OpenAI GPT-4o / GPT-4o-mini: high quality but adds API cost and dependency.
* **Rationale:** Groq’s `8b-instant` chosen to prioritize **fast ticket turnaround and cost efficiency**.

---

### 4) **Classification model**

* **Used:** Same Groq-hosted **`llama-3.1-8b-instant`**, prompted to output JSON conforming to a `TicketClassification` schema.
* **Why:**

  * One unified model handles both classification and generation (less infra complexity).
* **Alternatives considered:**

  * Dedicated small classifier model (e.g., fine-tuned DistilBERT). Would be cheaper per request but requires training + hosting overhead.
* **Rationale:** Using `llama-3.1-8b-instant` avoids maintaining a separate model, while still giving structured outputs reliably.

---

### 5) **Chunking & retrieval**

* **Used:** Sentence-Transformer embeddings split into \~800-token chunks with overlap, queried against Pinecone with `top_k=5`.
* **Trade-off:**

  * Small `k` → faster + cheaper, but may miss edge-case docs.
  * Larger `k` → more context, but higher token cost and slower.
* **Rationale:** `top_k=5` is a practical sweet spot for support ticket answers.

---

### 6) **Topic gating**

* **Policy:** Only run RAG if ticket tags intersect with:
  `{How-to, Product, Best practices, API/SDK, SSO}`.
* **Why:**

  * Reduces unnecessary RAG calls (saves cost, improves latency).
  * Keeps answers focused on documentation-backed queries.
* **Rationale:** Rule-based gating is cheap and transparent. If accuracy issues appear, can swap in an LLM-based classifier later.

---

➡️ So in short, pipeline is:

* **Vector DB:** Pinecone (`atlan-docs`)
* **Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
* **Generator:** Groq `llama-3.1-8b-instant`
* **Classifier:** Groq `llama-3.1-8b-instant`

---

## Repository layout

```
tickchat2/
├─ app.py                   # UI/API entrypoint (runs classifier + RAG/router)
├─ rag_pipeline.py          # Retrieval + prompt assembly + generation + citations
├─ crawler_to_pinecone.py   # Crawler + embedder + Pinecone upsert
├─ llm_utils.py             # Model wrappers (OpenAI / Groq / etc.)
├─ models.py                # Typed configs & response models
├─ requirements.txt         # Python dependencies
└─ README.md                # This file :)
```

*(Reflects the files visible in the repo listing.)* ([GitHub][1])

---

## Prerequisites

* **Python** 3.10+ (recommended)
* A **Pinecone** account & API key (or adapt for FAISS)
* An **LLM provider** key (OpenAI or Groq)
* (Optional) A `.env` file for secrets

---

## Quick start (local)

```bash
# 1) Clone
git clone https://github.com/anim2403/tickchat2.git
cd tickchat2

# 2) Create & activate a virtual environment
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate

# 3) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 4) Configure environment
cp .env.example .env   # if present; otherwise create .env (see below)

# 5) Build the index (crawl & ingest docs into Pinecone)
python crawler_to_pinecone.py \
  --roots https://docs.atlan.com/ https://developer.atlan.com/ \
  --index TICKCHAT2_INDEX \
  --namespace default \
  --max-pages 2000 \
  --concurrency 8 \
  --rps 2.0

# 6) Run the app locally
# If app.py is a Streamlit app:
streamlit run app.py
# If app.py is a simple FastAPI/Flask server:
# uvicorn app:app --reload --port 8000
```

> If you prefer a **local/offline** prototype, replace Pinecone with FAISS in `rag_pipeline.py` and the crawler (notes in [Extending](#extending)).

---

## Configuration

Create a `.env` in the project root (values shown are examples):

```ini
# LLM (choose one provider)
OPENAI_API_KEY=sk-********************************
# or
GROQ_API_KEY=gsk_*********************************

# Embeddings (choose one)
EMBEDDING_MODEL=openai:text-embedding-3-small
# or a local model id like: sentence-transformers/all-MiniLM-L6-v2

# Pinecone
PINECONE_API_KEY=********************************
PINECONE_INDEX=TICKCHAT2_INDEX
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=default

# Crawling
CRAWL_ROOTS=https://docs.atlan.com/,https://developer.atlan.com/
CRAWL_MAX_PAGES=2000
CRAWL_RPS=2.0

# App behavior
RAG_TOPICS=How-to,Product,Best practices,API/SDK,SSO
RETRIEVAL_K=6
MAX_TOKENS=1024
TEMPERATURE=0.2
```

*You can also pass many of these as CLI flags to the crawler.*

---

## Index build: crawl & ingest

Typical usage:

```bash
python crawler_to_pinecone.py \
  --roots https://docs.atlan.com/ https://developer.atlan.com/ \
  --index TICKCHAT2_INDEX \
  --namespace default \
  --backend sentence-transformers \
  --embedding-model all-MiniLM-L6-v2 \
  --chunk-size 800 \
  --chunk-overlap 120 \
  --max-pages 2000 \
  --concurrency 8 \
  --rps 2.0 \
  --include "docs.atlan.com/*" "developer.atlan.com/*" \
  --exclude "*.pdf" "*.zip" "*.png" "*.jpg" "*.svg"
```

**What it does**

* Respects `robots.txt` and rate limits.
* Extracts main text (strips nav, script, style).
* Splits into chunks with overlap.
* Embeds each chunk and **upserts** to Pinecone:
  `id, vector, {url, title, chunk_id, text[:N], ...}`.

---

## Run the app

Depending on how `app.py` is implemented:

### A) Streamlit UI (most common in these RAG demos)

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`).

### B) FastAPI/Flask server

```bash
uvicorn app:app --reload --port 8000
```

Visit `http://localhost:8000` (or use your preferred client).

**Environment variables** (from `.env`) control:

* which topics trigger RAG (`RAG_TOPICS`)
* model to use, retrieval depth, temperature
* Pinecone index/namespace

---

## How it works (runtime flow)

1. **Classify the ticket/question**

   * Simple **rule-based** check: if the detected topic ∈ `RAG_TOPICS`, proceed to RAG; otherwise return a routed message.
   * (Optional) Replace with a lightweight LLM classifier for more nuanced labels.

2. **Retrieve**

   * Vector search in Pinecone with `k=RETRIEVAL_K`.
   * Optionally apply **MMR** or **semantic re-rank** for diversity.

3. **Assemble prompt**

   * System + user message + **retrieved chunks** (titles/urls included).
   * Guardrails: “If the answer is not in the context, say you don’t know.”

4. **Generate**

   * Call the LLM via `llm_utils.py`.

5. **Cite sources**

   * Extract unique URLs from the retrieved set that influenced the answer.
   * Render them at the end of the response (bullet list or inline).

6. **Non-RAG topics**

   * Short confirmation like:

     > “This ticket has been classified as a ‘Connector’ issue and routed to the appropriate team.”

---

## Troubleshooting

* **No results / poor answers**

  * Increase `RETRIEVAL_K` to 8–12.
  * Increase `chunk-size` (e.g., 800–1200) with overlap (10–15%).
  * Verify the **index actually contains** your target domains (check the crawler logs).

* **Rate limits / 429s**

  * Lower `--rps` during crawl; add backoff/retry in `llm_utils.py`.

* **Token limit errors**

  * Reduce `RETRIEVAL_K` or chunk size; enable server-side truncation.

* **Pinecone auth errors**

  * Confirm `PINECONE_API_KEY`, `PINECONE_REGION`, and that the index exists.

* **Citations missing**

  * Ensure the retriever stores the `url` in metadata and the pipeline collects/render them.

---

## Extending

* **Switch to FAISS (local)**

  * Replace Pinecone client calls in `rag_pipeline.py` with FAISS index build/load.
  * Save vectors/metadata locally (e.g., `faiss.index`, `store.pkl`).
  * Good for offline demos or Streamlit Cloud with small corpora.

* **Classifier upgrade**

  * Swap the rule-based topic gate with a small model (e.g., `gpt-4o-mini`, `llama-3.1-8B-instruct`) that outputs one of your labels. Keep a fallback to the rules.

* **Re-ranking**

  * Add a cross-encoder re-rank step on the top 20 vectors before selecting the final 6–8 contexts.

* **Observability**

  * Log: question, top-k URLs, prompt tokens, output tokens, latency, provider.
  * Add a `/healthz` endpoint, Prometheus metrics, or Streamlit debug panel.

* **Incremental refresh**

  * Re-crawl on a schedule and upsert diffs (track content hash per URL).

---

## Security notes

* Never log raw API keys or entire prompts with user PII.
* If you support authenticated/private sources, **don’t embed secrets**; enforce allow-lists and scrub content during crawl.
* Consider a deny-list for domains and path patterns (CLI `--exclude` already helps).

---

## License

MIT (or your preferred OSS license).

---

### What I used from the repo

I relied on the public repo listing to capture file names and overall intent. If you want me to tailor this README to the **exact** code paths and CLI flags in your scripts, paste the contents of:

* `crawler_to_pinecone.py`
* `rag_pipeline.py`
* `app.py`
* `llm_utils.py`
* `models.py`
* `requirements.txt`

…and I’ll wire the docs precisely to your function/arg names and defaults.

[1]: https://github.com/anim2403/tickchat2/tree/main "GitHub - anim2403/tickchat2"
