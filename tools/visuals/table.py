import os
import uuid
import plotly.graph_objects as go
from typing import Optional

from mcp_server.types.visuals_type import TableImageData


def create_table_image(data: TableImageData, save_path: Optional[str] = "./.images") -> str:
    """Create and save a table image from structured data."""
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Create table with alternating row colors
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=data.headers,
            fill_color='#4CAF50',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=[list(row) for row in zip(*data.rows)],
            fill_color=[['#f2f2f2', 'white'] * (len(data.rows) // 2 + 1)],
            font=dict(size=10),
            align='center'
        )
    )])
    
    # Add title if provided
    if data.title:
        fig.update_layout(title=data.title)
    
    # Generate filename and save
    filename = f"table_{uuid.uuid4().hex[:8]}.png"
    file_path = os.path.join(save_path, filename)
    fig.write_image(file_path)
    
    return os.path.abspath(file_path)