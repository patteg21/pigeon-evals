from mcp.server.fastmcp import FastMCP

from mcp_server.clients import VectorDB, EmbeddingModel, SQLClient

from utils import logger
from mcp_server.types.visuals_type import (
    TableImageData,
    ChartData,
    BarChartData,
    FinancialChartData,
    DataPoint,
    FinancialDataPoint
)
from mcp_server.visuals.table import create_table_image
from mcp_server.visuals.charts import create_line_chart, create_bar_chart, create_financial_chart

def init_mcp_tools(mcp: FastMCP):
    vector_db_client: VectorDB = VectorDB()
    embedding_model: EmbeddingModel = EmbeddingModel(pca_path="artifacts/pca_512.joblib")
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

    @mcp.tool()
    def create_line_chart_visualization(
        data_points: list[dict],
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        line_style: str = "solid",
        color: str | None = None
    ) -> str:
        """
        Create a line chart visualization from data points.
        
        Args:
            data_points: List of dictionaries with 'x' and 'y' keys, optionally 'label'
            title: Optional title for the chart
            x_label: Optional label for x-axis
            y_label: Optional label for y-axis
            line_style: Line style: "solid", "dashed", or "dotted"
            color: Optional color for the line
            
        Returns:
            Absolute path to the generated PNG file
        """
        points = [DataPoint(**point) for point in data_points]
        chart_data = ChartData(
            data=points,
            title=title,
            x_label=x_label,
            y_label=y_label,
            line_style=line_style,
            color=color
        )
        return create_line_chart(chart_data)

    @mcp.tool()
    def create_bar_chart_visualization(
        categories: list[str],
        values: list[float],
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        colors: list[str] | None = None,
        horizontal: bool = False
    ) -> str:
        """
        Create a bar chart visualization.
        
        Args:
            categories: List of category names
            values: List of numeric values corresponding to categories
            title: Optional title for the chart
            x_label: Optional label for x-axis
            y_label: Optional label for y-axis
            colors: Optional list of colors for bars
            horizontal: Whether to create horizontal bars
            
        Returns:
            Absolute path to the generated PNG file
        """
        chart_data = BarChartData(
            categories=categories,
            values=values,
            title=title,
            x_label=x_label,
            y_label=y_label,
            colors=colors,
            horizontal=horizontal
        )
        return create_bar_chart(chart_data)

    @mcp.tool()
    def create_financial_chart_visualization(
        financial_data: list[dict],
        title: str | None = None,
        ticker: str | None = None,
        chart_type: str = "candlestick",
        show_volume: bool = True
    ) -> str:
        """
        Create a financial chart (candlestick/OHLC) visualization.
        
        Args:
            financial_data: List of dicts with 'date', 'open', 'high', 'low', 'close', optionally 'volume'
            title: Optional title for the chart
            ticker: Optional ticker symbol
            chart_type: Chart type: "candlestick", "ohlc", or "line"
            show_volume: Whether to show volume subplot
            
        Returns:
            Absolute path to the generated PNG file
        """
        points = [FinancialDataPoint(**point) for point in financial_data]
        chart_data = FinancialChartData(
            data=points,
            title=title,
            ticker=ticker,
            chart_type=chart_type,
            show_volume=show_volume
        )
        return create_financial_chart(chart_data)

