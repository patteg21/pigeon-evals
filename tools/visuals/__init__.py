from mcp.server.fastmcp import FastMCP


from utils.types.visuals_type import (
    FinancialChartData, 
    FinancialDataPoint, 
    TableImageData, 
    ChartData, 
    DataPoint,
    BarChartData
)
from evals.src.utils import logger
from evals.src.embedder import OpenAIEmbedder
from evals.src.storage.text import SQLiteDB

from .charts import create_bar_chart, create_financial_chart, create_line_chart
from .table import create_table_image


def init_visual_tools(mcp: FastMCP):
    embedding_model: OpenAIEmbedder = OpenAIEmbedder(pca_path="data/artifacts/pca_512.joblib")
    sql_client: SQLiteDB = SQLiteDB()
    
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

