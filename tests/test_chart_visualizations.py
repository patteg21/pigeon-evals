import os
import pytest
from datetime import datetime, timedelta

from mcp_server.visuals.charts import create_line_chart, create_bar_chart, create_financial_chart
from mcp_server.types.visuals_type import ChartData, BarChartData, FinancialChartData, DataPoint, FinancialDataPoint
from utils import logger


def test_line_chart_visualization():
    """Test the line chart visualization function"""
    
    # Create test data
    data_points = [
        DataPoint(x=1, y=10),
        DataPoint(x=2, y=15),
        DataPoint(x=3, y=13),
        DataPoint(x=4, y=17),
        DataPoint(x=5, y=20)
    ]
    
    chart_data = ChartData(
        data=data_points,
        title="Sales Growth Over Time",
        x_label="Quarter",
        y_label="Sales (millions)",
        line_style="solid",
        color="blue"
    )
    
    logger.info("Creating line chart visualization...")
    
    # Test line chart creation
    chart_path = create_line_chart(chart_data)
    logger.info(f"Line chart saved to: {chart_path}")
    
    # Verify file exists and is in .images directory
    if os.path.exists(chart_path):
        logger.info("✓ Line chart file created successfully")
        logger.info(f"File size: {os.path.getsize(chart_path)} bytes")
    else:
        logger.error("✗ Line chart file not found")
    
    assert os.path.exists(chart_path), "Line chart file should exist"
    assert chart_path.endswith('.png'), "Line chart should be PNG file"
    assert '.images' in chart_path, "Chart should be saved in .images directory"
    assert 'line_chart_' in os.path.basename(chart_path), "Filename should contain line_chart prefix"


def test_bar_chart_visualization():
    """Test the bar chart visualization function"""
    
    # Create test data
    chart_data = BarChartData(
        categories=["Q1", "Q2", "Q3", "Q4"],
        values=[100, 120, 135, 150],
        title="Quarterly Revenue",
        x_label="Quarter",
        y_label="Revenue (millions)",
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    )
    
    logger.info("Creating bar chart visualization...")
    
    # Test bar chart creation
    chart_path = create_bar_chart(chart_data)
    logger.info(f"Bar chart saved to: {chart_path}")
    
    # Verify file exists and is in .images directory
    if os.path.exists(chart_path):
        logger.info("✓ Bar chart file created successfully")
        logger.info(f"File size: {os.path.getsize(chart_path)} bytes")
    else:
        logger.error("✗ Bar chart file not found")
    
    assert os.path.exists(chart_path), "Bar chart file should exist"
    assert chart_path.endswith('.png'), "Bar chart should be PNG file"
    assert '.images' in chart_path, "Chart should be saved in .images directory"
    assert 'bar_chart_' in os.path.basename(chart_path), "Filename should contain bar_chart prefix"


def test_horizontal_bar_chart():
    """Test horizontal bar chart functionality"""
    
    chart_data = BarChartData(
        categories=["Product A", "Product B", "Product C"],
        values=[85, 92, 78],
        title="Product Performance",
        x_label="Score",
        y_label="Product",
        horizontal=True
    )
    
    logger.info("Creating horizontal bar chart...")
    
    chart_path = create_bar_chart(chart_data)
    logger.info(f"Horizontal bar chart saved to: {chart_path}")
    
    assert os.path.exists(chart_path), "Horizontal bar chart file should exist"
    assert '.images' in chart_path, "Chart should be saved in .images directory"
    assert 'bar_chart_' in os.path.basename(chart_path), "Filename should contain bar_chart prefix"


def test_financial_chart_candlestick():
    """Test candlestick financial chart visualization"""
    
    # Create test financial data
    base_date = datetime(2024, 1, 1)
    financial_data = []
    
    for i in range(10):
        date = base_date + timedelta(days=i)
        open_price = 100 + i * 2
        close_price = open_price + (-1 if i % 2 == 0 else 1) * (i % 3 + 1)
        high_price = max(open_price, close_price) + 2
        low_price = min(open_price, close_price) - 1
        volume = 1000000 + i * 100000
        
        financial_data.append(FinancialDataPoint(
            date=date,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        ))
    
    chart_data = FinancialChartData(
        data=financial_data,
        title="AAPL Stock Price",
        ticker="AAPL",
        chart_type="candlestick",
        show_volume=True
    )
    
    logger.info("Creating candlestick financial chart...")
    
    # Test financial chart creation
    chart_path = create_financial_chart(chart_data)
    logger.info(f"Financial chart saved to: {chart_path}")
    
    # Verify file exists and is in .images directory
    if os.path.exists(chart_path):
        logger.info("✓ Financial chart file created successfully")
        logger.info(f"File size: {os.path.getsize(chart_path)} bytes")
    else:
        logger.error("✗ Financial chart file not found")
    
    assert os.path.exists(chart_path), "Financial chart file should exist"
    assert chart_path.endswith('.png'), "Financial chart should be PNG file"
    assert '.images' in chart_path, "Chart should be saved in .images directory"
    assert 'financial_chart_' in os.path.basename(chart_path), "Filename should contain financial_chart prefix"


def test_financial_chart_ohlc():
    """Test OHLC financial chart visualization"""
    
    # Create simple test data
    financial_data = [
        FinancialDataPoint(
            date="2024-01-01",
            open=100,
            high=105,
            low=98,
            close=103,
            volume=1500000
        ),
        FinancialDataPoint(
            date="2024-01-02", 
            open=103,
            high=107,
            low=101,
            close=105,
            volume=1600000
        )
    ]
    
    chart_data = FinancialChartData(
        data=financial_data,
        title="Stock OHLC Chart",
        chart_type="ohlc",
        show_volume=False
    )
    
    logger.info("Creating OHLC financial chart...")
    
    chart_path = create_financial_chart(chart_data)
    logger.info(f"OHLC chart saved to: {chart_path}")
    
    assert os.path.exists(chart_path), "OHLC chart file should exist"
    assert '.images' in chart_path, "Chart should be saved in .images directory"
    assert 'financial_chart_' in os.path.basename(chart_path), "Filename should contain financial_chart prefix"


def test_financial_chart_line():
    """Test line style financial chart"""
    
    financial_data = [
        FinancialDataPoint(
            date="2024-01-01",
            open=100, high=105, low=98, close=103
        ),
        FinancialDataPoint(
            date="2024-01-02",
            open=103, high=107, low=101, close=105
        ),
        FinancialDataPoint(
            date="2024-01-03",
            open=105, high=108, low=104, close=107
        )
    ]
    
    chart_data = FinancialChartData(
        data=financial_data,
        title="Stock Price Trend",
        chart_type="line",
        show_volume=False
    )
    
    logger.info("Creating line financial chart...")
    
    chart_path = create_financial_chart(chart_data)
    logger.info(f"Line financial chart saved to: {chart_path}")
    
    assert os.path.exists(chart_path), "Line financial chart file should exist"
    assert '.images' in chart_path, "Chart should be saved in .images directory"
    assert 'financial_chart_' in os.path.basename(chart_path), "Filename should contain financial_chart prefix"


def test_line_chart_with_dates():
    """Test line chart with date x-axis"""
    
    data_points = [
        DataPoint(x="2024-01-01", y=100),
        DataPoint(x="2024-01-02", y=105),
        DataPoint(x="2024-01-03", y=103),
        DataPoint(x="2024-01-04", y=108),
        DataPoint(x="2024-01-05", y=112)
    ]
    
    chart_data = ChartData(
        data=data_points,
        title="Daily Values",
        x_label="Date",
        y_label="Value",
        line_style="dashed",
        color="red"
    )
    
    logger.info("Creating line chart with dates...")
    
    chart_path = create_line_chart(chart_data)
    logger.info(f"Date line chart saved to: {chart_path}")
    
    assert os.path.exists(chart_path), "Date line chart file should exist"
    assert '.images' in chart_path, "Chart should be saved in .images directory"
    assert 'line_chart_' in os.path.basename(chart_path), "Filename should contain line_chart prefix"