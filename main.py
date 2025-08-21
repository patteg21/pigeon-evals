import asyncio
import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from tools import init_mcp_tools
from evals.src.utils import logger

# loads env for local development
load_dotenv()

service: str | None = os.getenv("SERVICE")

async def main():
    mcp: FastMCP = FastMCP(name="Gareth Demo MCP")
    init_mcp_tools(mcp)

    # or streamable-http for http 
    if service == "http":
        logger.info("MCP Server Starting - HTTP ...")
        await mcp.run_streamable_http_async()
    else: 
        logger.info("MCP Server Starting - STDIO...")
        await mcp.run_stdio_async()
    

if __name__ == "__main__":
    asyncio.run(main()) 

