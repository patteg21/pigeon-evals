
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


# Preprocessing Pipeline (`evals/src`)

## Usage
**Note** that the usage uses PythonPath due to borrowing from the mcp_server just for development sake, typically these would live seperate.
```bash
PYTHONPATH=<absolute path to project> python evals/src/main.py --config evals/configs/test.yml
# or
PYTHONPATH=<absolute path to project> uv run evals/src/main.py --config evals/configs/test.yml
```

`PYTHONPATH=/Users/patteg/Desktop/development/gp-mcp-demo python evals/src/main.py --config evals/configs/test.yml`



#### Features
This is meant ot be a highly composable testing pipeline to be able to iterate and test as quickly as possible. The core of it uses templates so that I can quickly test different variations of parameters. All the pices of the flow are interchangable or removable, meaning I can add / remove / extend and functionality I need in future iterations. The main belief behind this architecture is so that I can keep developing and still consider all different ways we can process the data to get to an output.

All the core functionality and the YAML files are linked to `Pydantic` Models to enforce type checking so missing fields automcatically throw errors. This is important for some of the document processing as well because if it comes across an unexpected field, it will immediately throw and error. Theses will help to indicate logical failures in the pipeline where certain data fields are being improperly created. An additional step in the future would be more explicit checking as seem with `FormType` where we expect Two possible entries. We can take this a step further with `re` based matching to validate data that may be more complex.

Down the line I would ideally expand tests cases for each individual piece to the evals/src/ so that we can test each Object, from base models to the pieces of those base models.

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
  text_store: "sqlite"
  vector:
    upload: true    # if we can to upload the data
    clear: true     # if we want to clear the existing data in the index
    index: "sec-embedding"
  outputs: ["chunks", "documents"] # local outputs of items

# Below is not added but would be apart of the evaluation pipeline where we could provide use cases etc for the RAG to be tested and manually evaluated
generator: {provider: "openai", model: "gpt-4o-mini"}
judge:
  type: "llm"
  prompt: "You are a strict grader. Score 1-5 for relevance and faithfulness..."
  calibration: {gold_fraction: 0.1}
retrival: {type: "cosine", top_k: 10}

```



**Dimensionality Reduction**

Supports PCA for reducing embedding dimensions (configurable output dimensions). Ideally would add in alternatives for the future such as [UMAP](https://umap-learn.readthedocs.io/en/latest/) as well as several other methods.






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



#### MCP Tests Case Descriptions

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_mcp.py -v
python -m pytest tests/test_table_visualization.py -v
python -m pytest tests/test_chart_visualizations.py -v
python -m pytest tests/test_pca_loader.py -v
python -m pytest tests/test_agent_tool_usage.py -v
python -m pytest tests/test_vector_search_relevancy.py -v
```

- **`test_mcp.py`** - MCP server startup and tool registration
- **`test_table_visualization.py`** - Table image generation and file handling  
- **`test_chart_visualizations.py`** - Chart visualization tools (line, bar, financial charts using Plotly)
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