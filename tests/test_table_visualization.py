import os
import pytest

from mcp_server.visuals.table import create_table_image
from utils.typing import TableImageData
from utils import logger


def test_table_visualization():
    """Test the table visualization function directly"""
    
    # Create test data
    test_data = TableImageData(
        headers=["Company", "Revenue", "Profit", "Market Cap"],
        rows=[
            ["Apple", "$365B", "$95B", "$2.8T"],
            ["Microsoft", "$198B", "$61B", "$2.4T"],
            ["Google", "$282B", "$73B", "$1.7T"],
            ["Amazon", "$514B", "$33B", "$1.5T"]
        ],
        title="Tech Company Financial Comparison",
        caption="Data as of 2023 fiscal year"
    )
    
    logger.info("Creating table visualization...")
    
    # Test with default save path
    image_path = create_table_image(test_data)
    logger.info(f"Image saved to: {image_path}")
    
    # Verify file exists
    if os.path.exists(image_path):
        logger.info("✓ Image file created successfully")
        logger.info(f"File size: {os.path.getsize(image_path)} bytes")
    else:
        logger.info("✗ Image file not found")
    
    # Test with custom save path (also using images directory)
    custom_path = ".images"
    custom_image_path = create_table_image(test_data, save_path=custom_path)
    logger.info(f"Second image saved to: {custom_image_path}")
    
    if os.path.exists(custom_image_path):
        logger.info("✓ Second image created successfully")
    else:
        logger.info("✗ Second image file not found")


