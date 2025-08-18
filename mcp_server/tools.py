import numpy as np
from mcp.server.fastmcp import FastMCP

from mcp_server.clients import VectorDB, EmbeddingModel, SQLClient

from utils import logger
from utils.pca import PCALoader
from utils.typing import (
    EntityType,
    TableImageData,
)
from mcp_server.visuals.table import create_table_image

def init_mcp_tools(mcp: FastMCP):
    vector_db_client: VectorDB = VectorDB()
    embedding_model: EmbeddingModel = EmbeddingModel()
    sql_client: SQLClient = SQLClient()

    reducer: PCALoader | None = None
    try:
        reducer = PCALoader(path="artifacts/sec_pca_512.joblib").load()
        logger.info("PCA reducer loaded for query-time dimensionality reduction.")
    except Exception as e:
        logger.warning(f"PCA not loaded — using identity (no reduction). Reason: {e}")

    def _reduce(vec: list[float]) -> list[float]:
        if reducer and reducer.model is not None:
            return reducer.transform_one(vec)
        # identity + L2 normalize to keep cosine geometry stable if no PCA
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        return v.tolist()

    def _enrich_with_text(response):
        """Enrich search results with text content from SQLite"""
        if hasattr(response, 'matches'):
            for match in response.matches:
                metadata = match.get('metadata', {})
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
    async def vector_search(query: str, ticker: str | None = None):
        """
        Embed the query, apply PCA reducer (if loaded), then ANN search.
        """
        try:
            vec = await embedding_model.create_embedding(query)
            vec = _reduce(vec)
            response = vector_db_client.query(vec, top_k=10, include_metadata=True)
            response = _enrich_with_text(response)
            return response
        except Exception as e:
            logger.error(f"Error in vector_search: {e}")
            return {"error": str(e), "matches": []}


    @mcp.tool()
    async def search_on_metadata(
            query: str, 
            entity_type: EntityType | None = None, 
            year: str| None = None,
            ticker: str | None =None
        ):
        try:
            vector = await embedding_model.create_embedding(query)
            response = vector_db_client.retrieve_by_metadata(vector, entity_type=entity_type, year=year, ticker=ticker)
            response = _enrich_with_text(response)
            return response
        except Exception as e:
            logger.error(f"Error in search_on_metadata: {e}")
            return {"error": str(e), "matches": []}


    @mcp.tool()
    def search_by_id(vector_id: str):
        """This is meant for the Agent to Be able to Chain and `scroll` by content proximity"""
        try:
            response = vector_db_client.retrieve_from_id(vector_id=vector_id)
            # For single vector retrieval, we need to handle differently
            if response and hasattr(response, 'metadata'):
                document_id = response.metadata.get('document_id')
                if document_id:
                    try:
                        doc = sql_client.retrieve_document(document_id)
                        if doc:
                            response.metadata['text'] = doc['text']
                    except Exception as e:
                        logger.error(f"Failed to retrieve text for document {document_id}: {e}")
                        response.metadata['text'] = ""
            return response
        except Exception as e:
            logger.error(f"Error in search_by_id: {e}")
            return {"error": str(e)}


    @mcp.tool()
    def create_table_visualization(
        headers: list[str], 
        rows: list[list], 
        title: str | None = None,
        caption: str | None = None,
    ) -> str:
        """
        Create a table visualization image from data.
        
        Args:
            headers: List of column headers
            rows: List of rows, where each row is a list of cell values
            title: Optional title for the table
            caption: Optional caption below the table
            save_path: Optional custom path to save the image
            
        Returns:
            Absolute path to the generated image file
        """
        data = TableImageData(
            headers=headers,
            rows=rows,
            title=title,
            caption=caption
        )
        return create_table_image(data)

