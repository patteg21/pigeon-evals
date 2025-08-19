
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

# Overview

```bash
SEC Documents → 
Preprocessing Pipeline → 
Vector DB + SQL DB → 
MCP Server → 
Client/Agent
```

### Text Processing Features

1. **Fuzzy Pattern Matching**: Handles OCR artifacts and formatting inconsistencies
2. **Page Number Tracking**: Maintains document location context
3. **Chunk Overlap**: Preserves context across text boundaries
4. **Linked Structure**: Maintains prev/next relationships for navigation

### Metadata Extraction

Form-specific patterns extract:
- **10-K**: "For the fiscal year ended..."
- **10-Q**: "For the quarterly period ended..."
- **Commission File Numbers**: Standardized regulatory identifiers
- **Period End Dates**: Financial reporting periods

## Technology Stack

### Core Dependencies:
- **FastMCP**: MCP server framework
- **Pinecone**: Vector database service
- **OpenAI**: Embedding generation
- **Pydantic**: Data validation and serialization
- **SQLite**: Local text storage
- **scikit-learn**: PCA dimensionality reduction
- **Plotly**: Visualization generation


### Clients
SQL Lite : For Storing texts as some chunks were too large to be kept in the metadata of the Pinecone and this is also far more proper
Pinecone : Vector DB Store


### Preprocessing
To Run:
```bash
python preprocesssing.py 
# or
uv run preprocesssing.py
```

1. Read SEC filing text files from `data/` directory
2. Extract metadata using form-specific patterns
3. Parse table of contents and document structure
4. Extract and organize content by Parts and Items
5. Generate embeddings for each text chunk
6. Apply PCA dimensionality reduction (512 dimensions)
7. Store vectors in Pinecone and text content in SQLite

One of the most central pieces to this implementation is the use of Typing and Pydantic models in order to ensure consistency when creating Data. The end goal of this is to flag `null` or `None` values in the processing pipelines. Additional this allows code to be easy to understand and reference back to at later points if needed. Since the Code is centered around objects there is a higher focus on telemetry throughout the data processing pipeline. We can understand the data integretity and parsing ability by utilizing Pydantics Type Checking. When the pydantic throws and error, we become aware of a processing aware that would otherwise be allowed. **Additonally I added a cache feature to save some costs on API usage as I run the pipeline and test different pieces.**

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

### Tests

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

#### Test Case Descriptions

- **`test_mcp.py`** - MCP server startup and tool registration
- **`test_table_visualization.py`** - Table image generation and file handling  
- **`test_pca_loader.py`** - PCA model loading and 512-dimension reduction
- **`test_agent_tool_usage.py`** - Agent tool discovery and usage patterns
- **`test_vector_search_relevancy.py`** - AI agent evaluates search result quality
  - Tests 4 scenarios: Apple revenue, Microsoft Azure, Tesla production, general tech earnings
  - Scores semantic match, accuracy, completeness, context relevance (0.0-1.0)
  - Passes if ≥75% of cases score above threshold (0.6-0.7)


### TODO's

GENERAL
- **PCA Pipeline** : Currently I am using PCA to reduce dimensionality to fit in the VectorDB. I would prefer to use a non-linear dimensionality reduction like UMAP
- **MCP Tools** : The tooling is rather small and ideally I would have this also be able to interact with a Live API for more context beyond these documents. Additionally I would add in a few more graphs to display and instead store graphs in a Bucket Store (such as S3) which can handle individual Auth and display the images to users.
- **Architecture** : The project now acts like two server like entities. In an ideal world I would seperate them into different repos and follow package level structure for the preprocessing pipeline and follow server level python structure for the mcp server, which is more similiar to the current setup.
- **Error Handling**: While the pydantic models require that data fields be input in order to execute, I would take it a step further and add in some better error handling directly into the system with more `raise` error functions, already has some custom errors defined as well according to our pipeline. This applies to the MCP server as well, so users and agents can better handle interuptions rather than a python code error. Fo
**More Test Cases** : The current test framework is good as a foundation, but testing here can allow quicker iterations in the future with verifiable results
**Processing Pipeline** : I would make some refactors to the processing pipeline to adhere to better composible code standards, potentially a runner / workers architecture so it is more fault tolerant as well and easier to stop in place.
**Agent Context Windows**: Since there are a lot of different documents with different lengths adding in some functionality on the embedding search to handle potentially massive response if the top_k is large so the Agent does not get overwlehmed or OpenAI API call fails.
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