
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


# Document Processing Pipeline (`/evals/`)

## Overview
A modular pipeline for processing SEC documents with embeddings, storage, and evaluation. The pipeline transforms raw documents into searchable vector representations using configurable processors and storage backends. The primary driver behind this setup is to create a proper evaluation pipeline that can also operate in a production setting.

## Usage
```bash
PYTHONPATH=<absolute path to project> python evals/src/main.py --config evals/configs/test.yml
# or
PYTHONPATH=<absolute path to project> uv run evals/src/main.py --config evals/configs/test.yml
```

For example 
`PYTHONPATH=/Users/patteg/Desktop/development/gp-mcp-demo python evals/src/main.py --config evals/configs/test.yml`

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
task: sample
dataset_path: "data/"
processors: ["tables", "breaks"] # toc - short for Table of Contents Parses the TOC and seperates based on the different sections
embedding: 
  provider: "openai"
  model: "text-embedding-3-small"
  pooling_strategy: "mean"  # Other options:  max, weighted, smooth_decay
  dimension_reduction: {type: "PCA", dims: 512}   # This can be forgone, it is optional, Ideally I would add in UMAP / T-SNE though those are not implemented
  use_threading: true
  max_workers: 8

# Currently only supports the local sqlite and pinecone DB but ideally we can add more control on thinks like index
storage: 
  text_store: 
    client: "sqlite"
    upload: false
  vector:
    upload: true    # if we can to upload the data
    clear: true     # if we want to clear the existing data in the index
    index: "sec-embedding"
  outputs: ["chunks", "documents"] # local outputs of items

# Below is how to run multiple test cases for a given run
report:
  tests:
    - type: "agent"
      name: "AWS Earnings Test"
      prompt: "You are a helpful assistant Agent to discover more about the SEC Documnets in your tools"
      query: "Get me information on the latest earnings of AWS from 2024"
      mcp: 
        command: "uv"
        args:
          - "--directory"
          - "/Users/patteg/Desktop/development/gp-mcp-demo/"
          - "run"
          - "main.py"

    - type: "llm"
      name: "LLM Retrieval Judge"
      prompt: "You are a strict grader. Score 1-5 for relevance and faithfulness..."
      query: "Get me information on the latest revenue of AWS"
      retrieval: 
        top_k: 10

    - type: "human"
      name: "Sample Retrieval Results"
      query: "TSLA Earnings"
      retrieval: 
        top_k: 10
```

__TODO:__
- More ways to chunk the document, particularly around some of the markers within the document such as smaller sectional differences, bullet pointed lists, and paragraph level. Also adding in more versatility with overlapping, max_chunk_size etc 
- Include more Pydantic models and elmiate mores usages of dictionaries where reasonable for more structured and readble code practices. 
- Move the final parts of utils into more seperated pieces, primarily the typing which should be moved into the evals (since I am using shared typing I have not done so)
- Expirement with other types of search other than dense-embedding based, using sparse embeddings like TF-IDF or a mixture of both
- Finish the TOC Parser with the new updated code processing system (legacy code from first processing pipeline)
- If I had more access to the VectorDB I would also test some of the different sizing of Dimensions for Vectors

<br>
<br>

---


# MCP Server

FEATURES: 
- STDIO-compatible logging system
- Custom Error Exception Management System 


I opted to use the [Official MCP SDK](https://github.com/modelcontextprotocol/python-sdk) due to having used their Rust and TS servers in the past. The MCP can be implemented in both HTTP and STDIO. It is important to note that the project uses a custom logger instead of print statements due to the STDIO implementation.

To Run:
```bash
python main.py 
# or
uv run main.py
```

Tools
```bash
Search
    - vector_search
        Params: 
            query (str)
            ticker (Optional[str])
            entity_type (Optional[EntityType])
            year (Optional[str])

    - search_by_id
        Params:
            vector_id (str)

Visuals
    - create_table_visualization
        Params:
            headers (List[str])
            rows (List[List])
            title (Optional[str])
            caption (Optional[str])
    
    - create_line_chart_visualization
        Params:
            data_points (List[dict])  # [{"x": value, "y": value, "label": optional}]
            title (Optional[str])
            x_label (Optional[str])
            y_label (Optional[str])
            line_style (str)  # "solid", "dashed", "dotted"
            color (Optional[str])
    
    - create_bar_chart_visualization
        Params:
            categories (List[str])
            values (List[float])
            title (Optional[str])
            x_label (Optional[str])
            y_label (Optional[str])
            colors (Optional[List[str]])
            horizontal (bool)  # default: False
    
    - create_financial_chart_visualization
        Params:
            financial_data (List[dict])  # [{"date": str, "open": float, "high": float, "low": float, "close": float, "volume": optional}]
            title (Optional[str])
            ticker (Optional[str])
            chart_type (str)  # "candlestick", "ohlc", "line"
            show_volume (bool)  # default: True
```
