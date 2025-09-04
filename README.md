
# Pigeon Evals

**A comprehensive RAG (Retrieval-Augmented Generation) evaluation pipeline for document processing and retrieval system benchmarking**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UV](https://img.shields.io/badge/Package%20Manager-uv-purple.svg)](https://docs.astral.sh/uv/)

## Keywords
`RAG evaluation`, `retrieval augmented generation`, `document processing pipeline`, `vector embeddings`, `embedding evaluation`, `semantic search`, `vector database`, `Pinecone`, `Qdrant`, `OpenAI embeddings`, `document chunking`, `retrieval benchmarking`, `MCP server`, `evaluation framework`, `Python pipeline`, `text processing`, `machine learning`, `NLP`, `retrieval system`, `AI evaluation`, `LLM evaluation`, `human evaluation`

## What is Pigeon Evals?

Pigeon Evals is a **modular RAG evaluation framework** designed for comprehensive **document processing**, **retrieval system benchmarking**, and **LLM evaluation**. It provides end-to-end functionality from document ingestion to performance evaluation, with support for multiple storage backends, embedding providers, and evaluation methodologies.

### Perfect for:
- 🔍 **RAG system evaluation** and performance benchmarking  
- 📊 **Document processing pipeline** evaluation and optimization
- 📈 **Semantic search** and retrieval quality assessment
- 🤖 **AI agent evaluation** with MCP (Model Context Protocol) integration
- 🏗️ **Research and production** RAG system development
- 📚 **Vector database** experimentation and comparison
- 📋 **Multi-modal evaluation** (Human, LLM, and Agent-based)

## Environment

To install uv: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
curl install: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Pylance
Recommended install [Ruff](https://docs.astral.sh/ruff/) and set PyLance to typeChecking `strict` or `basic`


Creates a venv
```bash
uv venv
```

activate venv
```bash 
source .venv/bin/activate
```

To add dependencies + libraries
```bash
uv add <library>
```


# RAG Evaluation Pipeline (`/evals/`)

## Overview
A modular pipeline for processing documents with embeddings, storage, and comprehensive evaluation. The pipeline transforms raw documents into searchable vector representations using configurable processors and storage backends. The primary driver behind this setup is to create a proper RAG evaluation pipeline that can also operate in a production setting.

## Usage
```bash
python src/main.py --config evals/configs/test.yml
# or
uv run src/main.py --config evals/configs/test.yml
```

## Pipeline Architecture
The pipeline follows a sequential processing flow:

1. **Document Loading** - Reads raw documents from specified directory
2. **Text Processing** - Applies configurable processors (tables, page breaks, TOC parsing)
3. **Embedding Generation** - Creates vector representations using OpenAI embeddings
4. **Dimensionality Reduction** - Supports PCA for reducing embedding dimensions (configurable output dimensions). Ideally would add in alternatives for the future such as [UMAP](https://umap-learn.readthedocs.io/en/latest/) as well as several other methods.
5. **Storage** - Saves processed data to both text (SQLite) and vector (Pinecone) stores
6. **Report Generation** - Can run Human Eval, LLM Eval, and MCP Agent Eval pipelines dynamically based on runs or seperate from runs entirely. Currently works for my local mcp too!

### Key Features
- **Fully modular design** - Add/remove/swap any processing component
- **Pydantic validation** - Type-safe configuration with automatic error detection
- **Multi-threaded embedding** - Parallel processing for faster execution
- **Flexible storage** - Support for multiple text and vector storage backends
- **Flexible Framework** - Removing any piece of this does not break the system, it simply skips that part, meaning this can be a full end to end eval pipeline

```bash
task: "example_eval_task"

threading:
  max_workers: 8

preprocess:
  ocr: "easyocr"
parser:
  todo: "something"

embedding:
  provider: "huggingface"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  pooling_strategy: "mean"
  dimension_reduction:
    type: "PCA"
    dims: 256
  use_threading: true


storage:
  text_store:
    client: "sqlite"
    path: "./data/text_store.db"
    upload: true
  vector:
    upload: true
    clear: false
    index: "eval_index"
  outputs:
    - "chunks"
    - "documents"

eval:
  top_k: 10
  provider: "openai"
  model: "gpt-4o"
  evaluations: true
  metrics:
    - "ndcg"
    - "precision"
    - "recall"
  
  rerank:
    provider: "huggingface"
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: 5
  
  test:
    load:
      path: "data/tests/default.json"
      key: "tests"
      
    tests:
      - type: "llm"
        name: "basic_llm_test"
        query: "What is the main topic?"
        prompt: "Analyze the retrieved documents and provide a summary."
        eval_type:
          - "single"
      
      - type: "human"
        name: "human_review"
        query: "Quality assessment query"
      
      - type: "agent"
        name: "agent_test"
        query: "Test agent functionality"
        prompt: "Execute the agent task"
        mcp:
          command: "python"
          args:
            - "agent_script.py"
            - "--test"
```