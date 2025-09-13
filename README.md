

# tickchat2

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

## Design decisions & trade-offs

### 1) **Vector DB: Pinecone vs FAISS**

* **Picked:** Pinecone (managed, serverless, fast, easy to scale).
* **Trade-off:** External dependency + API cost vs. FAISS’s zero cost and local speed.
  *Guideline:* Use FAISS for fully offline or tiny datasets; use Pinecone for cloud deploys, sharing, and scale.

### 2) **Embeddings**

* **Options:** OpenAI `text-embedding-3-small` (cheap, strong quality), `text-embedding-3-large` (higher recall), or Sentence-Transformers (local).
* **Trade-off:** API dependency vs. offline control.
  *Guideline:* Start with `-3-small` for cost/quality balance; move up if recall is lacking.

### 3) **Chunking**

* **Approach:** Recursive splitter by tokens/chars with small overlap (e.g., 512–800 tokens, 10–15% overlap).
* **Trade-off:** Larger chunks increase recall but risk prompt bloat; smaller chunks reduce hallucinations but may lose context. Tune per corpus.

### 4) **Citations**

* **Decision:** Always show the **source URLs** used to build the answer.
* **Trade-off:** Slightly longer responses, but materially improves trust and debuggability.

### 5) **Topic gating (guardrails)**

* **Decision:** If topic ∈ {How-to, Product, Best practices, API/SDK, SSO} → **use RAG**. Otherwise → **route only** (no RAG).
* **Trade-off:** Simple logic is transparent and cheap vs. an LLM categorizer that’s more flexible but costs tokens. Start rule-based; upgrade to a lightweight LLM classifier if false positives/negatives appear.

### 6) **LLM provider**

* **Decision:** Keep the generator behind a small abstraction (`llm_utils.py`) so you can swap OpenAI/Groq/others.
* **Trade-off:** Slight indirection, but enables A/B testing and failover.

### 7) **Crawler**

* **Decision:** Cautious crawling (robots.txt aware, rate-limited, MIME filtered) with **deduping** and **canonical URLs**; store only clean text.
* **Trade-off:** Slower ingestion but cleaner KB, better retrieval quality.

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
