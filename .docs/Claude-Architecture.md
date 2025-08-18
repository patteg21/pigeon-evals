# Architecture Documentation

## Overview

This project is a comprehensive SEC filings analysis system that combines a robust preprocessing pipeline with a Model Context Protocol (MCP) server. The system processes SEC documents (10-K and 10-Q filings) from major tech companies, extracts structured data, and provides intelligent search capabilities through vector embeddings and semantic analysis.

## System Architecture

### High-Level Components

The system consists of two primary subsystems:

1. **Preprocessing Pipeline** - Processes raw SEC filings into structured, searchable data
2. **MCP Server** - Provides real-time query interface for processed data

### Data Flow Architecture

```
SEC Documents → Preprocessing Pipeline → Vector DB + SQL DB → MCP Server → Client/Agent
```

## Core Components

### 1. Preprocessing Pipeline (`preprocessing.py`)

The preprocessing pipeline is the foundation of the system, responsible for transforming raw SEC filing text into structured, searchable data.

#### Key Features:
- **Document Parsing**: Extracts structured content from SEC filings
- **Hierarchical Data Organization**: Documents → Parts → Items → Tables
- **Text Chunking**: Intelligently splits large documents while maintaining context
- **Embedding Generation**: Creates vector embeddings for semantic search
- **Dimensionality Reduction**: Uses PCA to optimize storage and search performance

#### Process Flow:
1. Read SEC filing text files from `data/` directory
2. Extract metadata using form-specific patterns
3. Parse table of contents and document structure
4. Extract and organize content by Parts and Items
5. Generate embeddings for each text chunk
6. Apply PCA dimensionality reduction (512 dimensions)
7. Store vectors in Pinecone and text content in SQLite

### 2. MCP Server (`main.py`, `mcp_server/`)

The MCP (Model Context Protocol) server provides a standardized interface for AI agents to interact with the processed SEC data.

#### Server Features:
- **Dual Protocol Support**: HTTP and STDIO communication protocols
- **FastMCP Framework**: Built on the official MCP SDK for reliability
- **Custom Logging**: STDIO-compatible logging system
- **Error Handling**: Comprehensive exception management

#### Available Tools:

##### Search Tools:
- **`vector_search`**: Semantic similarity search across all documents
- **`search_on_metadata`**: Filtered search by entity type, year, and ticker
- **`search_by_id`**: Direct retrieval by vector ID for chaining queries

##### Visualization Tools:
- **`create_table_visualization`**: Generates table images with custom styling

### 3. Data Storage Layer

#### Vector Database (Pinecone)
- **Purpose**: Stores document embeddings for semantic search
- **Index Configuration**: 512-dimensional vectors with cosine similarity
- **Metadata**: Structured metadata for filtering and enrichment
- **Features**: Fast ANN search, metadata filtering, scalable storage

#### SQL Database (SQLite)
- **Purpose**: Stores full text content that exceeds vector metadata limits
- **Schema**: Simple document table with ID-based retrieval
- **Integration**: Automatic text enrichment for search results
- **Benefits**: No size limitations, full text preservation

### 4. Data Models (`utils/typing/`)

The system uses strongly-typed Pydantic models to ensure data consistency and validation throughout the pipeline.

#### Core Models:

##### Document Hierarchy:
- **`SECDocument`**: Top-level document container
- **`SECPart`**: Document sections (Part I, II, III, IV)
- **`SECItem`**: Individual items within parts
- **`SECTable`**: Extracted tables and table of contents

##### Vector Models:
- **`VectorObject`**: Complete embedding container with metadata
- **`PineconeResponse`**: Query result wrapper with human-readable formatting

#### Type Safety Features:
- **Date Validation**: Ensures proper YYYY-MM-DD format
- **Entity Type Enforcement**: Controlled vocabulary for content types
- **Metadata Filtering**: Type-safe query construction
- **Automatic Year Extraction**: Derives year from date fields

### 5. Embedding and Dimensionality Reduction

#### Embedding Pipeline:
1. **Text Tokenization**: Intelligent chunking respecting document boundaries
2. **Embedding Generation**: OpenAI embeddings with pooling strategies
3. **PCA Reduction**: 512-dimensional vectors for optimal storage
4. **Normalization**: L2 normalization for stable cosine similarity

#### PCA Implementation (`utils/pca.py`):
- **Training**: One-time training on document corpus
- **Persistence**: Joblib serialization for consistent transforms
- **Query-time Application**: Real-time dimensionality reduction for searches
- **Fallback Handling**: Graceful degradation when PCA unavailable

### 6. Client Abstraction Layer (`mcp_server/clients/`)

#### VectorDB Client (`vector_db.py`):
- **Pinecone Integration**: Complete API wrapper with error handling
- **Metadata Queries**: Advanced filtering capabilities
- **Error Management**: Custom exception hierarchy
- **Upload Optimization**: Batch processing and validation

#### SQL Client (`sql_db.py`):
- **SQLite Abstraction**: Context-managed connections
- **Document Storage**: Efficient text storage and retrieval
- **Error Handling**: Comprehensive exception management
- **Connection Pooling**: Automatic resource management

#### Embedding Client (`embedding.py`):
- **OpenAI Integration**: Standardized embedding generation
- **Token Management**: Intelligent chunking and counting
- **Caching**: Disk-based embedding cache for efficiency
- **Strategy Support**: Multiple pooling strategies (mean, max, etc.)

## Data Processing Pipeline

### Document Structure Extraction

The system recognizes the hierarchical structure of SEC filings:

```
SECDocument
├── metadata (commission number, period end, etc.)
├── table_of_contents + all tables (SECTable)
├── parts[] (SECPart)
│   └── items[] (SECItem)
└── tables[] (SECTable)
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

### Development Tools:
- **pytest**: Testing framework
- **uv**: Package management
- **dotenv**: Environment configuration

## Configuration and Environment

### Required Environment Variables:
- `PINECONE_API_KEY`: Vector database authentication
- `OPENAI_API_KEY`: Embedding service authentication
- `SERVICE`: Server protocol selection (http/stdio)

### File Structure:
```
├── data/                    # SEC filing documents
├── artifacts/               # Trained models (PCA)
├── mcp_server/             # Server implementation
├── preprocess/             # Document processing
├── utils/                  # Utilities and type definitions
├── tests/                  # Test suite
└── images/                 # Generated visualizations
```

## Performance Considerations

### Optimization Strategies:
1. **PCA Dimensionality Reduction**: Reduces storage and query time
2. **Hybrid Storage**: Vectors in Pinecone, text in SQLite
3. **Embedding Caching**: Disk cache for repeated computations
4. **Batch Processing**: Efficient bulk operations
5. **Lazy Loading**: On-demand text retrieval

### Scalability Features:
- **Stateless Design**: Horizontal scaling capability
- **Database Separation**: Independent scaling of storage layers
- **Streaming Support**: Memory-efficient large file processing
- **Error Recovery**: Robust exception handling and retry logic

## Security and Reliability

### Data Protection:
- **Environment Variables**: Secure credential management
- **Input Validation**: Pydantic model enforcement
- **SQL Injection Prevention**: Parameterized queries
- **Error Boundary**: Graceful degradation on failures

### Monitoring and Observability:
- **Structured Logging**: JSON-formatted logs with context
- **Exception Tracking**: Detailed error reporting
- **Performance Metrics**: Embedding cache hit rates
- **Data Validation**: Real-time type checking

## Future Enhancements

### Planned Improvements:
1. **UMAP Integration**: Non-linear dimensionality reduction alternative
2. **Live API Integration**: Real-time SEC filing updates
3. **Advanced Visualizations**: Interactive charts and graphs
4. **Microservice Architecture**: Service separation for better scaling
5. **Enhanced Error Handling**: Custom exception hierarchy expansion
6. **Evaluation Framework**: Quality metrics and benchmarking
7. **Fault Tolerance**: Worker/runner architecture for processing pipeline

This architecture provides a robust foundation for SEC document analysis while maintaining flexibility for future enhancements and scaling requirements.