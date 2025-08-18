
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
python processsing 
# or
uv run processsing
```


One of the most central pieces to this implementation is the use of Typing and Pydantic models in order to ensure consistency when creating Data. The end goal of this is to flag `null` or `None` values in the processing pipelines. Additional this allows code to be easy to understand and reference back to at later points if needed. Since the Code is centered around objects there is a higher focus on telemetry throughout the data processing pipeline.

The Heirarchy of Objects Follows:
```bash
[SECDocumnet]
      |
      |
  [SECPart]
      |
      |
  [SECItem]
```

One of the other feaures that I began to implement is the way we capture metdata from the first few pages up until ther TOC. This gets less important though potential useful information such as the `commission_number` and `period_end`. Down the line ideally the metadata would also in clude some of the information found on these initial few pages. Some other metedata that is important to note about the pydantic objects. 
Metdata:
```bash
SECPart
    - Section
    - Page Number 

SECItem
    - Page
    - Page Number 
    - Subsection
```
Since these objects are essenitally a linked list, we have traceability throughout the heirarchy to any other metadata on a document / SECDocumnet.


One of core pieces of the way I preprocess is focused on the inherit structure within the documnets, particularly the TOC. This allows us to better Section & Partion the files rather a naive page break or naive chunking. By Utilizing the builtin `re` (Regex) library we can achieve some more siginificant seperation for our embedding models.

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