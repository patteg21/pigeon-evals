import os
from typing import Optional
from pathlib import Path
import uuid
import plotly.graph_objects as go

from utils.typing import TableImageData


def create_table_image(data: TableImageData, 
                      figsize: tuple[int, int] = (12, 8),
                      save_path: Optional[str] = "images") -> str:
    """
    Create a table image from structured data.
    
    Args:
        data: TableImageData containing headers, rows, and optional metadata
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the image file
        
    Returns:
        Path to the saved image file
    """
    os.makedirs(save_path, exist_ok=True)

    # Create alternating row colors
    num_rows = len(data.rows)
    row_colors = ['#f2f2f2' if i % 2 == 0 else 'white' for i in range(num_rows)]
    
    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=data.headers,
            fill_color='#4CAF50',
            font=dict(color='white', size=12),
            align='center',
            height=40
        ),
        cells=dict(
            values=[list(row) for row in zip(*data.rows)],
            fill_color=[row_colors] * len(data.headers),
            font=dict(size=10),
            align='center',
            height=30
        )
    )])
    
    # Update layout
    width, height = figsize[0] * 80, figsize[1] * 80  # Convert inches to pixels
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=60, b=40),
        font=dict(size=10)
    )
    
    # Add title if provided
    if data.title:
        fig.update_layout(title=dict(
            text=data.title,
            font=dict(size=16, color='black'),
            x=0.5
        ))
    
    # Add caption if provided
    if data.caption:
        fig.add_annotation(
            text=data.caption,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, style='italic'),
            xanchor='center'
        )
    
    # Generate save path if save_path is just a directory
    if save_path == "images" or (save_path and not save_path.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html'))):
        # Create images directory if it doesn't exist
        images_dir = Path(save_path)
        images_dir.mkdir(exist_ok=True)
        
        # Generate filename with UUID and optional title
        file_uuid = str(uuid.uuid4())
        if data.title:
            # Clean title for filename
            clean_title = "".join(c for c in data.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{clean_title.replace(' ', '_').lower()}_{file_uuid}.png"
        else:
            filename = f"table_{file_uuid}.png"
        
        save_path = str(images_dir / filename)
    
    # Save the image
    fig.write_image(save_path, width=width, height=height, scale=2)
    
    return os.path.abspath(save_path)