import numpy as np
from mcp.server.fastmcp import FastMCP

from clients import VectorDB, EmbeddingModel

from utils import logger
from utils.pca import PCALoader
from utils.typing import (
    EntityType,
    TableImageData
)
from mcp_server.visuals.table import create_table_image

def init_mcp_tools(mcp: FastMCP):
    vector_db_client: VectorDB = VectorDB()
    embedding_model: EmbeddingModel = EmbeddingModel()

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

    logger.info("Adding Tools to MCP Server...")

    @mcp.tool()
    async def vector_search(query: str, ticker: str | None = None):
        """
        Embed the query, apply PCA reducer (if loaded), then ANN search.
        """
        vec = await embedding_model.create_embedding(query)
        vec = _reduce(vec)
        return vector_db_client.query(vec, top_k=10, include_metadata=True)


    @mcp.tool()
    def search_metadata(
            query: str, 
            entity_type: EntityType | None = None, 
            year: str| None = None,
            ticker: str | None =None
        ):

        vector = embedding_model.create_embedding(query)

        vector_db_client.retrieve_by_metadata(vector, entity_type=entity_type, year=year, ticker=ticker)


    @mcp.tool()
    def search_by_id(vector_id: str):
        """This is meant for the Agent to Be able to Chain and `scroll` through related content proximity"""
        vector_db_client.retrieve_from_id(vector_id=vector_id)

    @mcp.tool()
    def latest_filing(query: str, year: int, ticker: str | None):
        pass

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

    