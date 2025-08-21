from mcp.server.fastmcp import FastMCP


from .search import init_search_tools
from .visuals import init_visual_tools



def init_mcp_tools(mcp: FastMCP):
    """
    A Shared point to init all different tooling capabilities


    Args:
        mcp (FastMCP): MCP Server
    """


    init_search_tools(mcp)
    init_visual_tools(mcp)


