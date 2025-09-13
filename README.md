
# TICKCHAT

An end-to-end, production-lean RAG (Retrieval-Augmented Generation) assistant for classifying support tickets and answering “product knowledge” questions using a searchable knowledge base built from public documentation.

* **Routable topics** (e.g., *How-to, Product, Best practices, API/SDK, SSO*) → go through the RAG pipeline and return an answer with citations.
* **Everything else** → display a clear, routed message (e.g., “This ticket has been classified as a ‘Connector’ issue and routed to the appropriate team.”)

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

---

## Why this project?

Support bots are often either:

* **LLM-only** (fluent but can hallucinate), or
* **Search-only** (accurate but verbose, not conversational).

This project combines **document retrieval** with **generative answers** so users get concise, cited, up-to-date responses for the topics that matter, and clean routing for everything else.

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

* `crawler_to_pinecone.py`: crawls allowed sites, cleans & chunks text, embeds, and **upserts into Pinecone**.
* `rag_pipeline.py`: wraps retrieval + prompt assembly + generation + citation formatting.
* `llm_utils.py`: LLM wrapper for **classification + generation**. Returns **topic tags with confidence scores**, sentiment, priority, and core problem.
* `models.py`: Pydantic classes for ticket classification and response typing.
* `app.py`: minimal UI/API that classifies topics and either:

  * calls RAG (for *How-to, Product, Best practices, API/SDK, SSO*), or
  * returns a routed classification message for everything else.

---

## Design decisions & trade-offs

### Vector database

* **Used:** [Pinecone](https://www.pinecone.io/)
* **Why:** Managed, scalable, no local persistence issues on Streamlit/Replit.

### Embedding model

* **Used:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
* **Why:** Fast, zero-cost, widely used for semantic search.

### Generator (answering)

* **Used:** Groq-hosted `llama-3.1-8b-instant`.
* **Why:** Low latency + good balance between cost and reasoning.

### Classification model

* **Used:** Same `llama-3.1-8b-instant`.
* **Output:** JSON with:

  * `topic_tags` (with **confidence scores**)
  * `sentiment`
  * `priority`
  * `core_problem`

Confidence scores are essential because they make routing decisions transparent and allow filtering low-confidence classifications.

### Topic gating

* **Policy:** Run RAG only if tags intersect with `{How-to, Product, Best practices, API/SDK, SSO}`.
* **Why:** Saves cost + latency, ensures retrieval is used only where docs exist.

---

## Repository layout

```
tickchat2/
├─ app.py                   # UI/API entrypoint
├─ rag_pipeline.py          # Retrieval + prompt assembly + generation
├─ crawler_to_pinecone.py   # Crawl + embed + upsert to Pinecone
├─ llm_utils.py             # LLM classification & generation (with confidence scores)
├─ models.py                # Typed data classes (topic tags, confidence, sentiment, priority, etc.)
├─ requirements.txt         # Python dependencies
└─ README.md
```

---

## Prerequisites

* Python 3.10+
* Pinecone API key
* Groq API key
* Optional: `.env` for secrets

---

## Quick start (local)

```bash
git clone https://github.com/anim2403/tickchat2.git
cd tickchat2

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

pip install --upgrade pip
pip install -r requirements.txt

# Crawl docs & ingest into Pinecone
python crawler_to_pinecone.py \
  --roots https://docs.atlan.com/ https://developer.atlan.com/ \
  --index TICKCHAT2_INDEX \
  --namespace default \
  --max-pages 2000 \
  --concurrency 8 \
  --rps 2.0

# Run app
streamlit run app.py
```

---

## Configuration

Example `.env`:

```ini
# LLM
GROQ_API_KEY=gsk_*********************************

# Pinecone
PINECONE_API_KEY=********************************
PINECONE_INDEX=atlan-docs
NAMESPACE=atlan

# Crawler
CRAWL_ROOTS=https://docs.atlan.com/,https://developer.atlan.com/

# App behavior
RAG_TOPICS=How-to,Product,Best practices,API/SDK,SSO
RETRIEVAL_K=6
```

---

## Index build: crawl & ingest

Respects `robots.txt`, cleans text, chunks, embeds, and upserts to Pinecone with metadata (url, title, text preview).

---

## Run the app

```bash
streamlit run app.py
```

---

## How it works (runtime flow)

1. **Classify the ticket**

   * `llm_utils` calls Groq model.
   * Returns `topic_tags` **with confidence scores**, `sentiment`, `priority`, `core_problem`.

   **Example JSON output:**

   ```json
   {
     "topic_tags": [
       {"tag": "Product", "confidence": 0.92},
       {"tag": "How-to", "confidence": 0.81},
       {"tag": "Connector", "confidence": 0.34}
     ],
     "sentiment": "neutral",
     "priority": "medium",
     "core_problem": "User is asking how to configure API authentication"
   }
   ```

2. **Check topic gating**

   * If tags ∈ `{How-to, Product, Best practices, API/SDK, SSO}`, → run RAG.
   * Else, → return routed message.

3. **Retrieve**

   * Pinecone vector search (`k=RETRIEVAL_K`).

4. **Assemble prompt**

   * Include system msg + user msg + retrieved chunks.

5. **Generate answer**

   * LLM produces response.

6. **Cite sources**

   * Unique URLs from retrieved docs shown in final answer.

---

