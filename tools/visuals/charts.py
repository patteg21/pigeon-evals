import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import uuid
from typing import Optional

from utils.types.visuals_type import (
    ChartData,
    BarChartData, 
    FinancialChartData,
)
from evals.src.utils import logger

def create_line_chart(data: ChartData, save_path: Optional[str] = "./.images") -> str:
    """
    Create a line chart visualization from ChartData using Plotly.
    
    Args:
        data: ChartData object containing chart configuration
        save_path: Optional directory path to save the image (default: ./.images)
        
    Returns:
        Absolute path to the generated PNG file
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Generate filename
    filename = f"line_chart_{uuid.uuid4().hex[:8]}.png"
    file_path = os.path.join(save_path, filename)
    
    # Extract x and y values
    x_vals = [point.x for point in data.data]
    y_vals = [point.y for point in data.data]
    
    # Handle datetime conversion if needed
    if x_vals and isinstance(x_vals[0], str):
        try:
            x_vals = pd.to_datetime(x_vals)
        except:
            pass
    
    # Create line chart
    fig = go.Figure()
    
    line_style_map = {'solid': 'solid', 'dashed': 'dash', 'dotted': 'dot'}
    line_style = line_style_map.get(data.line_style, 'solid')
    color = data.color or 'blue'
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        line=dict(color=color, dash=line_style, width=2),
        marker=dict(size=6),
        name=data.title or 'Data'
    ))
    
    # Update layout
    fig.update_layout(
        title=data.title or 'Line Chart',
        xaxis_title=data.x_label or 'X',
        yaxis_title=data.y_label or 'Y',
        template='plotly_white',
        showlegend=False
    )
    
    # Save as PNG
    fig.write_image(file_path)
    
    logger.info(f"Line chart saved to: {file_path}")
    return os.path.abspath(file_path)

def create_bar_chart(data: BarChartData, save_path: Optional[str] = "./.images") -> str:
    """
    Create a bar chart visualization from BarChartData using Plotly.
    
    Args:
        data: BarChartData object containing chart configuration
        save_path: Optional directory path to save the image (default: ./.images)
        
    Returns:
        Absolute path to the generated PNG file
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Generate filename
    filename = f"bar_chart_{uuid.uuid4().hex[:8]}.png"
    file_path = os.path.join(save_path, filename)
    
    fig = go.Figure()
    
    # Set up colors
    colors = data.colors or ['steelblue'] * len(data.categories)
    if len(colors) < len(data.categories):
        colors = colors * (len(data.categories) // len(colors) + 1)
    colors = colors[:len(data.categories)]
    
    # Create bar chart
    if data.horizontal:
        fig.add_trace(go.Bar(
            y=data.categories,
            x=data.values,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.2f}' for v in data.values],
            textposition='outside'
        ))
    else:
        fig.add_trace(go.Bar(
            x=data.categories,
            y=data.values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in data.values],
            textposition='outside'
        ))
    
    # Update layout
    fig.update_layout(
        title=data.title or 'Bar Chart',
        xaxis_title=data.x_label or 'Categories',
        yaxis_title=data.y_label or 'Values',
        template='plotly_white',
        showlegend=False
    )
    
    # Save as PNG
    fig.write_image(file_path)
    
    logger.info(f"Bar chart saved to: {file_path}")
    return os.path.abspath(file_path)

def create_financial_chart(data: FinancialChartData, save_path: Optional[str] = "./.images") -> str:
    """
    Create a financial chart (candlestick/OHLC) from FinancialChartData using Plotly.
    
    Args:
        data: FinancialChartData object containing chart configuration
        save_path: Optional directory path to save the image (default: ./.images)
        
    Returns:
        Absolute path to the generated PNG file
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Generate filename
    filename = f"financial_chart_{uuid.uuid4().hex[:8]}.png"
    file_path = os.path.join(save_path, filename)
    
    # Convert data to DataFrame for easier handling
    df_data = []
    for point in data.data:
        date = point.date
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        df_data.append({
            'date': date,
            'open': point.open,
            'high': point.high,
            'low': point.low,
            'close': point.close,
            'volume': point.volume or 0
        })
    
    df = pd.DataFrame(df_data)
    
    # Create subplots if volume is shown
    if data.show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price', 'Volume')
        )
    else:
        fig = go.Figure()
    
    # Create price chart based on type
    if data.chart_type == "candlestick":
        candlestick = go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )
        if data.show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
    
    elif data.chart_type == "ohlc":
        ohlc = go.Ohlc(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )
        if data.show_volume:
            fig.add_trace(ohlc, row=1, col=1)
        else:
            fig.add_trace(ohlc)
    
    else:  # line chart
        line = go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Close Price'
        )
        if data.show_volume:
            fig.add_trace(line, row=1, col=1)
        else:
            fig.add_trace(line)
    
    # Add volume chart if requested
    if data.show_volume:
        # Color volume bars based on price movement
        volume_colors = ['green' if close >= open else 'red' 
                        for open, close in zip(df['open'], df['close'])]
        
        volume_bar = go.Bar(
            x=df['date'],
            y=df['volume'],
            marker_color=volume_colors,
            name='Volume',
            opacity=0.7
        )
        fig.add_trace(volume_bar, row=2, col=1)
    
    # Update layout
    title = data.title or f"{data.ticker or 'Stock'} Financial Chart"
    
    if data.show_volume:
        fig.update_layout(
            title=title,
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=600
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
    else:
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            height=500
        )
    
    # Save as PNG
    fig.write_image(file_path)
    
    logger.info(f"Financial chart saved to: {file_path}")
    return os.path.abspath(file_path)