# RAG with LangChain & Pinecone

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-000000?logo=pinecone&logoColor=white)

> Retrieval-Augmented Generation pipeline: document ingestion, vector indexing with Pinecone, and context-aware generation using OpenAI.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Theoretical Background](#theoretical-background)
- [RAG Pipeline Details](#rag-pipeline-details)
- [AWS SageMaker Execution Evidence](#aws-sagemaker-execution-evidence)
- [References](#references)

---

## Overview

This repository implements a complete **Retrieval-Augmented Generation (RAG)** pipeline. The system ingests a web document, chunks it, stores embeddings in **Pinecone**, and uses an **LLM agent** to answer queries with retrieved context.

> [!IMPORTANT]
> This is the second of two repositories for Lab 04. The first repo covers LangChain basics; this repo builds the **full RAG system** with Pinecone as the vector store.

| Aspect | Description |
|:-------|:------------|
| **Domain** | Information Retrieval & NLP |
| **Task** | Retrieval-Augmented Generation |
| **LLM** | OpenAI `gpt-4o-mini` |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536 dims) |
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
                    │  OpenAI Embeddings               │
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
                    │  @tool retrieve_context()        │
                    │    │                             │
                    │    ▼                             │
                    │  similarity_search(query, k=2)   │
                    │    │                             │
                    │    ▼                             │
                    │  LLM Agent (gpt-4o-mini)         │
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
| **Indexing** | Embed | `OpenAIEmbeddings` | Converts text → 1536-dim vectors |
| **Indexing** | Store | `PineconeVectorStore` | Persists embeddings in Pinecone |
| **Retrieval** | Search | `similarity_search()` | Cosine similarity, top-k results |
| **Generation** | Answer | `create_agent()` | LLM with retrieval tool |

---

## Setup & Installation

### Prerequisites

- **Python 3.11+**
- **Jupyter Notebook / Lab**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
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
pinecone>=5.0.0
beautifulsoup4>=4.12.0
```

</details>

### API Key Setup

The notebook uses `getpass` to securely prompt for API keys at runtime:

| Key | Required For | How to Get |
|:----|:-------------|:-----------|
| `OPENAI_API_KEY` | LLM + Embeddings | [platform.openai.com](https://platform.openai.com/api-keys) |
| `PINECONE_API_KEY` | Vector Store | [app.pinecone.io](https://app.pinecone.io/) |

> [!TIP]
> No keys are stored in the code. The notebook prompts you at execution time.

### Run the Notebook

```bash
jupyter notebook rag_langchain_pinecone.ipynb
```

---

## Notebook Walkthrough

| # | Section | Description | Key API |
|:-:|:--------|:------------|:--------|
| 1 | Setup | Install packages, configure API keys | `getpass` |
| 2 | Components | Initialize LLM + Embeddings | `init_chat_model`, `OpenAIEmbeddings` |
| 3 | Pinecone Config | Create/connect serverless index | `Pinecone`, `ServerlessSpec` |
| 4.1 | Load Documents | Fetch blog post via web scraping | `WebBaseLoader`, `BeautifulSoup` |
| 4.2 | Split Documents | Chunk into 1000-char segments | `RecursiveCharacterTextSplitter` |
| 4.3 | Store Vectors | Index all chunks in Pinecone | `PineconeVectorStore.add_documents()` |
| 5.1 | Similarity Search | Direct vector search demo | `similarity_search()` |
| 5.2 | RAG Agent | Create agent with retrieval tool | `@tool`, `create_agent()` |
| 5.3 | Query Demo | Ask questions about the blog post | `agent.stream()` |
| 6 | Cleanup | Optional: delete Pinecone index | `pc.delete_index()` |

---

## Theoretical Background

### What is RAG?

**Retrieval-Augmented Generation** combines two capabilities:

1. **Retrieval** — Find relevant documents from a knowledge base using semantic search.
2. **Generation** — Use an LLM to synthesize an answer, grounding it in the retrieved context.

> [!NOTE]
> RAG addresses the key limitation of LLMs: their knowledge cutoff. By augmenting prompts with real-time retrieved data, responses are factual and up-to-date.

### Why Pinecone?

Pinecone is a managed vector database optimized for similarity search:

| Feature | Benefit |
|:--------|:--------|
| **Serverless** | No infrastructure management |
| **Cosine Similarity** | Effective for text embeddings |
| **Scalable** | Handles millions of vectors |
| **Low Latency** | Sub-100ms queries |

### Embedding Dimensions

| Model | Dimensions | Use Case |
|:------|:-----------|:---------|
| `text-embedding-3-small` | 1536 | Cost-efficient, general purpose |
| `text-embedding-3-large` | 3072 | Higher accuracy, more expensive |

> [!IMPORTANT]
> The Pinecone index dimension **must match** the embedding model dimension. This project uses `text-embedding-3-small` (1536 dims).

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

### Agent Pattern

The RAG agent uses LangChain's **tool-calling** pattern:

1. User sends a query
2. Agent decides to call `retrieve_context` tool
3. Tool performs `similarity_search(query, k=2)` on Pinecone
4. Results are returned as `ToolMessage` with content + artifacts
5. Agent synthesizes a final response using the retrieved context

---

## AWS SageMaker Execution Evidence

> [!NOTE]
> This section contains evidence of successful notebook execution on AWS SageMaker.

### Deployment Steps

1. Navigate to AWS SageMaker Studio
2. Create a new Notebook Instance
3. Upload `rag_langchain_pinecone.ipynb` and `src/` directory
4. Select Python 3 (Data Science) kernel
5. Provide API keys when prompted
6. Run all cells

<!-- Screenshots to be added after SageMaker execution -->

---

## References

1. LangChain Documentation. [Build a RAG Agent](https://python.langchain.com/docs/tutorials/rag/).
2. LangChain Documentation. [Pinecone Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone).
3. Weng, L. (2023). [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/).
4. Pinecone Documentation. [Quickstart Guide](https://docs.pinecone.io/guides/get-started/quickstart).
5. OpenAI. [Embeddings Guide](https://platform.openai.com/docs/guides/embeddings).

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
