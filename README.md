
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


# Preprocessing Pipeline (`evals/`)

The preprocessing pipeline is a modular, configurable system for processing SEC documents and generating embeddings for vector search. The pipeline follows a document → chunks → embeddings → storage flow.

## Architecture

The pipeline consists of four main stages managed by runners:

1. **Data Loading** (`loader/data_loader.py`) - Loads SEC documents from the `data/` directory
2. **Processing** (`runner/processor_runner.py`) - Extracts chunks using configurable processors
3. **Embedding** (`runner/embedder_runner.py`) - Generates embeddings with optional dimensionality reduction
4. **Storage** (`runner/storage_runner.py`) - Stores results in vector database and SQLite

## Available Processors

- **breaks** - Splits documents at page breaks and logical sections
- **tables** - Extracts and processes tabular data 
- **toc** - Processes table of contents structures

## Embedding Providers

- **openai** - OpenAI text-embedding models (e.g., text-embedding-3-small)
- **huggingface** - HuggingFace transformer models

## Configuration

Pipeline behavior is controlled via YAML config files in `evals/configs/`:

```yaml
task: sample
dataset_path: "data/"
sec_metadata: ["commission_number"]
processors: ["tables", "breaks", "toc"]
embedding: 
  provider: "openai"
  model: "text-embedding-3-small"
  pooling_strategy: "mean"  # mean, max, weighted, smooth_decay
  dimension_reduction: {type: "PCA", dims: 512}   # UMAP / T-SNE
  use_threading: true
  max_workers: 8

storage: 
  text_store: "sqlite"
  vector:
    upload: false
    clear: false
    index: "sec-embedding"
  outputs: ["chunks"]

report:
  output_path: "evals/reports/sample"
```

## Usage

```bash
PYTHONPATH=<absolute path to project> python evals/src/main.py --config evals/configs/test.yml
# or
PYTHONPATH=<absolute path to project> uv run evals/src/main.py --config evals/configs/test.yml
```

## Dimensionality Reduction

Supports PCA for reducing embedding dimensions (configurable output dimensions). Also considered [UMAP](https://umap-learn.readthedocs.io/en/latest/) for non-linear dimensionality reduction.



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
    - search_on_metadata
        Params:
            query (str)
            entity_type (Optional[EntityType])
            year (Optional[str])
            ticker (Optional[str])
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
```



#### MCP Tests Case Descriptions

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_mcp.py -v
python -m pytest tests/test_table_visualization.py -v
python -m pytest tests/test_pca_loader.py -v
python -m pytest tests/test_agent_tool_usage.py -v
python -m pytest tests/test_vector_search_relevancy.py -v
```

- **`test_mcp.py`** - MCP server startup and tool registration
- **`test_table_visualization.py`** - Table image generation and file handling  
- **`test_pca_loader.py`** - PCA model loading and 512-dimension reduction
- **`test_agent_tool_usage.py`** - Agent tool discovery and usage patterns
- **`test_vector_search_relevancy.py`** - AI agent evaluates search result quality
  - Tests 4 scenarios: Apple revenue, Microsoft Azure, Tesla production, general tech earnings
  - Scores semantic match, accuracy, completeness, context relevance (0.0-1.0)
  - Passes if ≥75% of cases score above threshold (0.6-0.7)


### TODO's

**Testing** - Add more tests to the eval pipeline
**Remove Legacy Code** - There are some parts of the codebase that I migrated away from and need to work
**Auto Run Naming + Tracking** - In my evals, auto creating different runs and names for runs to better segregate testing and put trained PCA / T-SNE / UMAP models in them
**Auto Eval** - Implementing a LLM Judge Model into the processing pipeline similiar to what exists in the Test case
**SEC Data Parsing** - Implement some REGEX to auto collect some basic information and attach to the document, such as Commission number
**MCP Tools** - Add more tools to the MCP Server as it is rather limited currently