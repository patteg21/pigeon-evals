
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
To Run:
```bash
python preprocesssing.py 
# or
uv run preprocesssing.py
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

### Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_mcp.py -v
python -m pytest tests/test_table_visualization.py -v
python -m pytest tests/test_pca_loader.py -v
python -m pytest tests/test_agent_tool_usage.py -v
```


### TODO's

GENERAL
- **PCA Pipeline** : Currently I am using PCA to reduce dimensionality to fit in the VectorDB. I would prefer to use a non-linear dimensionality reduction like UMAP
- **MCP Tools** : The tooling is rather small and ideally I would have this also be able to interact with a Live API for more context beyond these documents. Additionally I would add in a few more graphs to display and instead store graphs in a Bucket Store (such as S3) which can handle individual Auth and display the images to users.
- **Architecture** : The project now acts like two server like entities. In an ideal world I would seperate them into different repos and follow package level structure for the preprocessing pipeline and follow server level python structure for the mcp server, which is more similiar to the current setup.
- **Error Handling**: While the pydantic models require that data fields be input in order to execute, I would take it a step further and add in some better error handling directly into the system with more `raise` error functions with some custom errors defined as well according to our pipeline. This applies to the MCP server as well, so users and agents can better handle interuptions rather than a python code error.
**More Test Cases** : The current test framework is good as a foundation, but testing here can allow quicker iterations in the future with verifiable results
**Processing Pipeline** : I would make some refactors to the processing pipeline to adhere to better composible code standards, potentially a runner / workers architecture so it is more fault tolerant as well and easier to stop in place.
**Evaluation** : 

SPECIFIC
- **Vector DB Client** (`mcp_server/clients/vector_db.py:29`): Add error handling for non-existent years
- **Document Extraction** (`preprocess/extraction.py:38`): Improve the naive search for table of contents
- **OCR Processing** (`preprocess/extraction.py:70`): Fix the OCR or handle with REGEX
- **MCP Testing** (`tests/test_mcp.py:25`): Update dummy system prompt
- **MCP Examples** (`tests/test_mcp.py:42`): Add more examples to showcase system capabilities
- **Metadata Extraction** (`preprocess/metadata.py:10`): Get more metadata such as various Q&A
- **Code Refactoring** (`preprocessing.py:130`): Refactor into more composable code
- **Large File Handling** (`preprocessing.py:153`): Handle very large files and chunk them due to metadata limitations
- **Dimensionality Reduction** (`preprocessing.py:277`): Consider UMAP as alternative to PCA for Pinecone index size of 512