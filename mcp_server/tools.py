from mcp.server.fastmcp import FastMCP

from utils import logger

def init_mcp_tools(mcp: FastMCP):
    
    logger.info("Adding Tools to MCP Server...")

    @mcp.tool()
    def vector_search(query: str, ticker: str | None):
        pass

    @mcp.tool()
    def latest_filing(query: str ):
        pass

    