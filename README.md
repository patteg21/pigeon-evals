
## Environment

To install uv: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
curl install: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Pylance
Recommended install Ruff and set PyLance to typeChecking `strict` or `basic`


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

# Overview

### Preprocessing
All logic for preprocessing in contained within the processsing/ folder with `__main__.py` to run. This folder borrows the clients and typing from the utils.
```bash
python processsing.py 
# or
uv run processsing.py
```

One of the most central pieces to this implementation is the use of Typing and Pydantic models in order to ensure consistency when creating Data. The end goal of this is to flag `null` or `None` values in the processing pipelines. Additional this allows code to be easy to understand and reference back to at later points if needed. Since the Code is centered around objects there is a higher focus on telemetry throughout the data processing pipeline. We can understand the data integretity and parsing ability by utilizing Pydantics Type Checking. When the pydantic throws and error, we become aware of a processing aware that would otherwise be allowed.

The Heirarchy of Objects Follows:
```bash
[SECDocumnet] -- [SECTable]
      |
      |
  [SECPart]
      |
      |
  [SECItem]
```

Dimesionality Reduction using PCA but also considered (Uniform Manifold Approximation and Projection - UMAP)[https://umap-learn.readthedocs.io/en/latest/].



### MCP Server
I opted to use the [Official MCP SDK](https://github.com/modelcontextprotocol/python-sdk) due to having used their Rust and TS servers in the past. The MCP can be implemented in both HTTP and STDIO. It is important to note that the project uses a custom logger instead of print statements due to the STDIO implementation.


Tools
```bash
Search
    - Vector Search
        Params: 
            Year (Optional[Int])
            Ticker (Optional[str])

Visuals
    - Bar Chart
        Params:
            Data (List[Dict]) 
    - Graph
        Params:
            Data (List[Dict]) 
```

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_mcp.py -v
python -m pytest tests/test_table_visualization.py -v
python -m pytest tests/test_pca_loader.py -v
```