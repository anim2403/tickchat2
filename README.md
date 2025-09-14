
# TICKCHAT

[Live app](https://tickchat2-dlpfscev8wgdx6wpjarhdr.streamlit.app/)

[Demo_Video](https://drive.google.com/file/d/1cT_pulXzbOrYYy7l-RXrMpAOhlskjNcU/view?usp=sharing)

An end-to-end, production-lean RAG (Retrieval-Augmented Generation) assistant for classifying support tickets and answering “product knowledge” questions using a searchable knowledge base built from public documentation.

* **Routable topics** (e.g., *How-to, Product, Best practices, API/SDK, SSO*) → go through the RAG pipeline and return an answer **with citations**.
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

* `crawler_to_pinecone.py`: Crawls allowed sites, cleans & chunks text, embeds, and **upserts into Pinecone**.
* `rag_pipeline.py`: Retrieval + prompt assembly + answer generation + citation formatting.
* `llm_utils.py`: LLM wrapper for **classification + generation**. Returns **topic tags with confidence scores**, sentiment, priority, and core problem.
* `models.py`: Pydantic classes for ticket classification and response typing.
* `app.py`: Streamlit app that classifies topics and either:

  * Calls RAG (for *How-to, Product, Best practices, API/SDK, SSO*), or
  * Returns a routed classification message for everything else.

---

## Design decisions & trade-offs

### Vector database

* **Used:** [Pinecone](https://www.pinecone.io/)
* **Why:** Managed, scalable, avoids persistence issues on Streamlit/Replit.

### Embedding model

* **Used:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
* **Why:** Fast, zero-cost, widely used for semantic search.

### Generator (answering)

* **Used:** Groq-hosted `llama-3.1-8b-instant`.
* **Why:** Low latency + good balance between cost and reasoning.

### Classification model

* **Used:** Same `llama-3.1-8b-instant`.
* **Output:** JSON with:

  * `topic_tags` (**with confidence scores**)
  * `sentiment`
  * `priority`
  * `core_problem`

Confidence scores make routing decisions transparent and allow filtering out low-confidence predictions.

### Topic gating

* **Policy:** Run RAG only if tags intersect with `{How-to, Product, Best practices, API/SDK, SSO}`.
* **Why:** Saves cost + latency, ensures retrieval is only used where docs exist.

---

## Repository layout

```
tickchat2/
├─ app.py                   # UI entrypoint (Streamlit)
├─ rag_pipeline.py          # Retrieval + prompt assembly + generation
├─ crawler_to_pinecone.py   # Crawl + embed + upsert to Pinecone
├─ llm_utils.py             # Classification & generation logic (with confidence scores)
├─ models.py                # Pydantic schemas (topic tags, confidence, sentiment, priority, etc.)
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
  --index atlan-docs \
  --namespace atlan \
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
PINECONE_NAMESPACE=atlan

# Crawler
CRAWL_ROOTS=https://docs.atlan.com/,https://developer.atlan.com/

# App behavior
RAG_TOPICS=How-to,Product,Best practices,API/SDK,SSO
RETRIEVAL_K=5
MAX_TOKENS=800
TEMPERATURE=0.1
```

---

## Index build: crawl & ingest

* Respects `robots.txt` and rate limits.
* Cleans text, strips nav/scripts/styles.
* Splits into overlapping chunks.
* Embeds each chunk and upserts to Pinecone with metadata (`url`, `title`, `text preview`, `chunk_id`).

---

## Run the app

```bash
streamlit run app.py
```

---

## How it works (runtime flow)

1. **Classify the ticket**

   * `llm_utils` calls Groq model.
   * Returns `topic_tags` **with confidence scores**, `sentiment`, `priority`, and `core_problem`.

   **Example JSON output:**

   ```json
   {
     "topic_tags": {
       "Product": 0.92,
       "How-to": 0.81,
       "Connector": 0.34
     },
     "sentiment": "Neutral",
     "priority": "P1",
     "core_problem": "User is asking how to configure API authentication"
   }
   ```

   *Tags with confidence < 0.6 are filtered out in the pipeline.*

2. **Check topic gating**

   * If any tag ∈ `{How-to, Product, Best practices, API/SDK, SSO}` → run RAG.
   * Else → return routed message.

3. **Retrieve**

   * Pinecone vector search (`k=RETRIEVAL_K`).

4. **Assemble prompt**

   * System msg + user msg + retrieved chunks.

5. **Generate answer**

   * Groq model produces structured, grounded answer.

6. **Cite sources**

   * Unique URLs from retrieved docs shown at the end of the answer.

---


