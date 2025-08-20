from mcp.server.fastmcp import FastMCP

from shared.clients import VectorDB, EmbeddingModel, SQLClient

from evals.src.utils import logger

def init_search_tools(mcp: FastMCP):
    vector_db_client: VectorDB = VectorDB()
    embedding_model: EmbeddingModel = EmbeddingModel(pca_path="data/artifacts/pca_512.joblib")
    sql_client: SQLClient = SQLClient()
    
    if embedding_model.pca_reducer:
        logger.info("PCA reducer loaded for query-time dimensionality reduction.")
    else:
        logger.warning("PCA not loaded — using identity (no reduction).")

    def _enrich_with_text(response):
        """Enrich search results with text content from SQLite"""
        if hasattr(response, 'matches'):
            for match in response.matches:
                # Handle Pinecone match object properly
                metadata = getattr(match, 'metadata', {}) or {}
                document_id = metadata.get('document_id')
                if document_id:
                    try:
                        doc = sql_client.retrieve_document(document_id)
                        if doc:
                            metadata['text'] = doc['text']
                    except Exception as e:
                        logger.error(f"Failed to retrieve text for document {document_id}: {e}")
                        metadata['text'] = ""
        return response


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
                response = vector_db_client.query(vec, top_k=top_k, include_metadata=True, filter=filters)
            else:
                response = vector_db_client.query(vec, top_k=top_k, include_metadata=True)
            response = _enrich_with_text(response)
            
            # Convert Pinecone response to JSON-serializable format
            result = {
                "matches": []
            }
            
            if hasattr(response, 'matches') and response.matches:
                for match in response.matches:
                    metadata = getattr(match, 'metadata', {}) or {}
                    
                    # Get full text from SQL database using document_id from metadata
                    document_id = metadata.get('document_id')
                    if document_id:
                        try:
                            doc = sql_client.retrieve_document(document_id)
                            if doc and doc.get('text'):
                                metadata['text'] = doc['text']
                        except Exception as e:
                            logger.error(f"Failed to retrieve text for document {document_id}: {e}")
                            # Keep existing text if retrieval fails
                    
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
            response = vector_db_client.retrieve_from_id(vector_id=vector_id)
            
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

