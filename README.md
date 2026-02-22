# RAG with LangChain & Pinecone

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-000000?logo=pinecone&logoColor=white)

> Retrieval-Augmented Generation pipeline: document ingestion, vector indexing with Pinecone, and context-aware generation using GPT-4o-mini via GitHub Models.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Execution Results](#execution-results)
- [Theoretical Background](#theoretical-background)
- [RAG Pipeline Details](#rag-pipeline-details)
- [References](#references)

---

## Overview

This repository implements a complete **Retrieval-Augmented Generation (RAG)** pipeline. The system ingests a web document, chunks it, stores embeddings in **Pinecone**, and uses an **LCEL RAG chain** to answer queries with retrieved context.

> [!IMPORTANT]
> This is the second of two repositories for Lab 04. The first repo covers LangChain basics; this repo builds the **full RAG system** with Pinecone as the vector store.

| Aspect | Description |
|:-------|:------------|
| **Domain** | Information Retrieval & NLP |
| **Task** | Retrieval-Augmented Generation |
| **LLM** | `gpt-4o-mini` via GitHub Models (FREE) |
| **Embeddings** | `all-MiniLM-L6-v2` local HuggingFace (384 dims, FREE) |
| **Vector Store** | Pinecone Serverless (AWS `us-east-1`) |
| **Data Source** | [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — Lilian Weng |

---

## Project Structure

```text
/
├── README.md
├── .gitignore
├── rag_langchain_pinecone.ipynb
└── src/
    ├── lab04-moodle.md
    └── requirements.txt
```

---

## Architecture

The RAG pipeline consists of two distinct phases:

```
                    ┌─────────────────────────────────┐
                    │       INDEXING PHASE (Offline)   │
                    │                                  │
                    │  Web URL                         │
                    │    │                             │
                    │    ▼                             │
                    │  WebBaseLoader + BeautifulSoup   │
                    │    │                             │
                    │    ▼                             │
                    │  RecursiveCharacterTextSplitter  │
                    │    │                             │
                    │    ▼                             │
                    │  HuggingFace Embeddings (Local)  │
                    │    │                             │
                    │    ▼                             │
                    │  Pinecone Vector Store           │
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │    RETRIEVAL PHASE (Runtime)     │
                    │                                  │
                    │  User Query                      │
                    │    │                             │
                    │    ▼                             │
                    │  Retriever (similarity, k=3)     │
                    │    │                             │
                    │    ▼                             │
                    │  LCEL RAG Chain                  │
                    │  (retriever | prompt | LLM)      │
                    │    │                             │
                    │    ▼                             │
                    │  Context-Aware Response          │
                    └─────────────────────────────────┘
```

### Component Breakdown

| Phase | Step | Component | Purpose |
|:------|:-----|:----------|:--------|
| **Indexing** | Load | `WebBaseLoader` | Fetches HTML, filters with BS4 |
| **Indexing** | Split | `RecursiveCharacterTextSplitter` | 1000-char chunks, 200 overlap |
| **Indexing** | Embed | `HuggingFaceEmbeddings` | Converts text → 384-dim vectors (local) |
| **Indexing** | Store | `PineconeVectorStore` | Persists embeddings in Pinecone |
| **Retrieval** | Search | `as_retriever(k=3)` | Cosine similarity, top-k results |
| **Generation** | Answer | LCEL RAG Chain | `retriever | prompt | model | parser` |

---

## Setup & Installation

### Prerequisites

- **Python 3.11+**
- **Jupyter Notebook / Lab**
- **GitHub Token** (Fine-grained PAT for GitHub Models — FREE)
- **Pinecone API Key** ([Get one here](https://app.pinecone.io/))

### Install Dependencies

```bash
pip install -r src/requirements.txt
```

<details>
<summary>📦 requirements.txt contents</summary>

```
langchain>=0.3.0
langchain-openai>=0.3.0
langchain-pinecone>=0.2.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-huggingface>=0.1.0
pinecone>=5.0.0
beautifulsoup4>=4.12.0
sentence-transformers>=3.0.0
```

</details>

### API Key Setup

The notebook uses `getpass` to securely prompt for keys at runtime:

| Key | Required For | How to Get |
|:----|:-------------|:-----------|
| `GITHUB_TOKEN` | LLM (gpt-4o-mini) | [github.com/settings/tokens](https://github.com/settings/tokens) |
| `PINECONE_API_KEY` | Vector Store | [app.pinecone.io](https://app.pinecone.io/) |

> [!TIP]
> No OpenAI API key is needed. The LLM is accessed for FREE via GitHub Models using your GitHub token.

### Run the Notebook

```bash
jupyter notebook rag_langchain_pinecone.ipynb
```

---

## Notebook Walkthrough

| # | Section | Description | Key API |
|:-:|:--------|:------------|:--------|
| 1 | Setup | Install packages, configure API keys | `getpass` |
| 2 | Components | Initialize LLM + Embeddings | `ChatOpenAI`, `HuggingFaceEmbeddings` |
| 3 | Pinecone Config | Create/connect serverless index (384 dims) | `Pinecone`, `ServerlessSpec` |
| 4.1 | Load Documents | Fetch blog post via web scraping | `WebBaseLoader`, `BeautifulSoup` |
| 4.2 | Split Documents | Chunk into 1000-char segments | `RecursiveCharacterTextSplitter` |
| 4.3 | Store Vectors | Index all chunks in Pinecone | `PineconeVectorStore.add_documents()` |
| 5.1 | Similarity Search | Direct vector search demo | `similarity_search()` |
| 5.2 | RAG Chain | Build LCEL chain with retriever | `RunnablePassthrough`, `StrOutputParser` |
| 5.3 | Query Demo | Ask questions about the blog post | `rag_chain.invoke()` |
| 6 | Cleanup | Optional: delete Pinecone index | `pc.delete_index()` |

---

## Execution Results

All cells executed successfully with Python 3.11.

### Cell Outputs Summary

| Cell | Output |
|:-----|:-------|
| **2. Components** | `LLM: GitHub Models (gpt-4o-mini)` · `Embeddings: Local HuggingFace (all-MiniLM-L6-v2)` |
| **3. Pinecone** | `Index already exists: arep-lab04-rag-local` · `Vector store ready.` |
| **4.1 Load** | `Loaded 1 document(s)` · `Total characters: 43047` |
| **4.2 Split** | `Split into 63 chunks` |
| **4.3 Store** | `Indexed 63 documents in Pinecone` |
| **5.1 Search** | 3 results returned for *"What is task decomposition?"* |
| **5.2 RAG Chain** | `RAG chain ready.` |
| **5.3 Query** | See below |

### Query Demo Output

**Query:** *"What is task decomposition?"*

**Response:**
> Task decomposition is the process of breaking down a larger task into smaller, manageable sub-tasks or steps. This can be done in several ways, including:
> 1. Using a language model (LLM) with simple prompting, such as asking for steps or subgoals for achieving a specific task.
> 2. Providing task-specific instructions, like asking for a story outline when writing a novel.
> 3. Involving human inputs to guide the breakdown of the task.
>
> Additionally, there is an approach known as LLM+P, which involves using an external classical planner for long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) to describe the planning problem, where the LLM translates the problem into PDDL, requests a planner to generate a PDDL plan, and then translates the plan back into natural language.

---

## Theoretical Background

### What is RAG?

**Retrieval-Augmented Generation** combines two capabilities:

1. **Retrieval** — Find relevant documents from a knowledge base using semantic search.
2. **Generation** — Use an LLM to synthesize an answer, grounding it in the retrieved context.

> [!NOTE]
> RAG addresses the key limitation of LLMs: their knowledge cutoff. By augmenting prompts with real-time retrieved data, responses are factual and up-to-date.

### Why Pinecone?

| Feature | Benefit |
|:--------|:--------|
| **Serverless** | No infrastructure management |
| **Cosine Similarity** | Effective for text embeddings |
| **Scalable** | Handles millions of vectors |
| **Low Latency** | Sub-100ms queries |

### Embedding Model

| Model | Dimensions | Cost | Use Case |
|:------|:-----------|:-----|:---------|
| `all-MiniLM-L6-v2` | 384 | FREE (local) | Lightweight, fast, good accuracy |

> [!IMPORTANT]
> The Pinecone index dimension **must match** the embedding model dimension. This project uses `all-MiniLM-L6-v2` (384 dims).

---

## RAG Pipeline Details

### Document Source

The pipeline indexes Lilian Weng's blog post [*LLM Powered Autonomous Agents*](https://lilianweng.github.io/posts/2023-06-23-agent/) — a comprehensive article covering planning, memory, and tool-use in LLM agents.

### Chunking Strategy

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| `chunk_size` | 1000 chars | Fits within context window |
| `chunk_overlap` | 200 chars | Preserves context at boundaries |
| `add_start_index` | `True` | Enables source tracking |

### RAG Chain Pattern (LCEL)

The RAG chain uses LangChain's **Expression Language (LCEL)**:

1. User sends a query
2. `retriever` performs `similarity_search(query, k=3)` on Pinecone
3. `format_docs` joins retrieved documents into a context string
4. `ChatPromptTemplate` builds the prompt with context + question
5. `model` (GPT-4o-mini) generates the response
6. `StrOutputParser` extracts the final text

---

## References

1. LangChain Documentation. [Build a RAG App](https://python.langchain.com/docs/tutorials/rag/).
2. LangChain Documentation. [Pinecone Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone).
3. Weng, L. (2023). [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/).
4. Pinecone Documentation. [Quickstart Guide](https://docs.pinecone.io/guides/get-started/quickstart).
5. HuggingFace. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

---

## Author

**Sergio Andrey Silva Rodriguez**  
*Systems Engineering Student*  
Escuela Colombiana de Ingeniería Julio Garavito

---

<details>
<summary>License</summary>

This project is for educational purposes as part of the AREP course at Escuela Colombiana de Ingenieria Julio Garavito.

</details>
