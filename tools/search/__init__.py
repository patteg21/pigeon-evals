from mcp.server.fastmcp import FastMCP

from evals.src.storage.vector import PineconeDB
from evals.src.storage.text import SQLiteDB

from evals.src.embedder import OpenAIEmbedder

from evals.src.utils import logger

def init_search_tools(mcp: FastMCP):
    pinecone_client: PineconeDB = PineconeDB()
    embedding_model: OpenAIEmbedder = OpenAIEmbedder(pca_path="data/artifacts/pca_512.joblib")
    sql_client: SQLiteDB = SQLiteDB()
    

    logger.info("Adding Tools to MCP Server...")

    @mcp.tool()
    async def vector_search(query: str, ticker: str | None = None, year: str | None = None, top_k: int =3):
        """
        Embed the query, apply PCA reducer (if loaded), then ANN search.
        """
        try:
            vec = await embedding_model.create_pinecone_embeddings(query)
            
            # Build metadata filters
            filters = {}
            if ticker:
                filters["ticker"] = {"$eq": ticker.upper()}
            if year:
                # Handle both string and numeric year values
                try:
                    year_float = float(year)
                    filters["year"] = {"$eq": year_float}
                except ValueError:
                    logger.warning(f"Invalid year format: {year}")
            
            # Query with filters if any are specified
            if filters:
                response = pinecone_client.query(vec, top_k=top_k, include_metadata=True, filter=filters)
            else:
                response = pinecone_client.query(vec, top_k=top_k, include_metadata=True)
            
            # Convert Pinecone response to JSON-serializable format and enrich with SQL text
            result = {
                "matches": []
            }
            
            if hasattr(response, 'matches') and response.matches:
                for match in response.matches:
                    metadata = getattr(match, 'metadata', {}) or {}
                    
                    # Replace any existing text with full text from SQL database using document_id
                    document_id = metadata.get('document_id')
                    if document_id:
                        try:
                            doc = sql_client.retrieve_document(document_id)
                            if doc and doc.get('text'):
                                metadata['text'] = doc['text']  # Replace with full text from SQL
                            else:
                                metadata['text'] = ""  # Clear if no text found
                        except Exception as e:
                            logger.error(f"Failed to retrieve text for document {document_id}: {e}")
                            metadata['text'] = ""  # Clear on error
                    
                    match_dict = {
                        "id": getattr(match, 'id', None),
                        "score": getattr(match, 'score', None),
                        "metadata": metadata
                    }
                    result["matches"].append(match_dict)
            
            return result
        except Exception as e:
            logger.error(f"Error in vector_search: {e}")
            return {"error": str(e), "matches": []}




    @mcp.tool()
    def search_by_id(vector_id: str):
        """This is meant for the Agent to Be able to Chain and `scroll` by content proximity"""
        try:
            response = pinecone_client.retrieve_from_id(vector_id=vector_id)
            
            # Convert to JSON-serializable format without embeddings
            result = {
                "id": getattr(response, 'id', None),
                "metadata": getattr(response, 'metadata', {}) or {}
            }
            
            # Get full text from SQL database using document_id from metadata
            document_id = result["metadata"].get('document_id')
            if document_id:
                try:
                    doc = sql_client.retrieve_document(document_id)
                    if doc and doc.get('text'):
                        result["metadata"]['text'] = doc['text']
                except Exception as e:
                    logger.error(f"Failed to retrieve text for document {document_id}: {e}")
                    result["metadata"]['text'] = ""
            
            return result
        except Exception as e:
            logger.error(f"Error in search_by_id: {e}")
            return {"error": str(e)}

