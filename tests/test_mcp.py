import os
import asyncio

from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from agents import Agent, Runner, set_default_openai_key

from utils import logger

load_dotenv()

openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
	raise EnvironmentError("OPENAI_API_KEY not set...")
set_default_openai_key(openai_api_key) 

async def run(server):
	"""
		Runs the OpenAI Agent with the provided MCP server

		Args:
			server (MCPServerStdio): The MCP server to use
	"""
	
	# TODO: Dummy system prompt, feel free to modify if needed
	agent = Agent(
		name="Assistant", 
		instructions=f"You are a helpful financial analyst assistant. Use the tools to answer questions based on SEC filings.",
		mcp_servers=[server],
	)

	# Test table visualization
	table_message = "Create a table visualization with headers ['Company', 'Revenue', 'Profit'] and rows [['Apple', '$365B', '$95B'], ['Microsoft', '$198B', '$61B'], ['Google', '$282B', '$73B']] with title 'Tech Company Financials'"
	logger.info("Testing table visualization...")
	table_result = await Runner.run(starting_agent=agent, input=table_message)
	logger.info("Table visualization result:", table_result.final_output)

	message = "What is the most recent revenue reported by Apple?"
	result = await Runner.run(starting_agent=agent, input=message)
	logger.info(result.final_output)

	# TODO: Add more examples to showcase the capabilities of the system

async def test():
	"""
		Defines the MCP server and runs the OpenAI Agent
	"""
	
	# Set up MCP server parameters to run the main.py script
	params = {
		"command": "python",
		"args": ["main.py"]
	}
	async with MCPServerStdio(
		params=params
	) as server:
		await run(server)

asyncio.run(test())
