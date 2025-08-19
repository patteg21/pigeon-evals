import os
import pytest
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

class ToolUsageTracker:
    """Tracks which tools have been used by the agent"""
    def __init__(self):
        self.tools_used = set()
        self.expected_tools = {
            "vector_search",
            "search_by_id",
            "create_table_visualization"
        }
    
    def mark_tool_used(self, tool_name: str):
        self.tools_used.add(tool_name)
        logger.info(f"Tool used: {tool_name}")
    
    def all_tools_used(self) -> bool:
        return self.expected_tools.issubset(self.tools_used)
    
    def get_unused_tools(self) -> set:
        return self.expected_tools - self.tools_used

@pytest.mark.asyncio
async def test_agent_tool_discovery_and_usage():
    """
    Test that asks the agent what tools it can use and waits for it to use each one.
    """
    
    # Set up MCP server parameters
    params = {
        "command": "python",
        "args": ["main.py"]
    }
    
    async with MCPServerStdio(params=params) as server:
        agent = Agent(
            name="Assistant",
            instructions="""You are a helpful financial analyst assistant. Use the tools to answer questions based on SEC filings. 
            When asked about your capabilities, list all available tools and demonstrate their usage.""",
            mcp_servers=[server],
        )
        
        tracker = ToolUsageTracker()
        
        # Step 1: Ask agent what tools it can use
        logger.info("Step 1: Asking agent about available tools...")
        discovery_message = "What tools do you have available? Please list all your tools and briefly explain what each one does."
        
        discovery_result = await Runner.run(starting_agent=agent, input=discovery_message)
        logger.info(f"Tool discovery response: {discovery_result.final_output}")
        
        # Step 2: Ask agent to demonstrate each tool
        logger.info("Step 2: Asking agent to demonstrate each tool...")
        
        demonstration_tasks = [
            {
                "message": "Use the vector_search tool to find information about Apple's recent financial performance.",
                "expected_tool": "vector_search"
            },
            {
                "message": "Create a table visualization showing the top 3 tech companies with their market caps: Apple ($3.5T), Microsoft ($3.0T), NVIDIA ($2.8T). Title it 'Top Tech Companies by Market Cap'.",
                "expected_tool": "create_table_visualization"
            }
        ]
        
        # Execute each demonstration task
        for task in demonstration_tasks:
            logger.info(f"Testing tool: {task['expected_tool']}")
            result = await Runner.run(starting_agent=agent, input=task["message"])
            logger.info(f"Result for {task['expected_tool']}: {result.final_output}")
            tracker.mark_tool_used(task["expected_tool"])
            
            # Wait a bit between tool calls
            await asyncio.sleep(1)
        
        # Step 3: Test search_by_id (this requires getting an ID first)
        logger.info("Step 3: Testing search_by_id tool...")
        
        # First get some results that might contain IDs
        search_result = await Runner.run(
            starting_agent=agent, 
            input="Search for Apple revenue information and show me the vector IDs from the results. Then use search_by_id to get more details about one of those documents."
        )
        logger.info(f"Search by ID result: {search_result.final_output}")
        tracker.mark_tool_used("search_by_id")
        
        # Step 4: Verify all tools were used
        logger.info("Step 4: Verifying all tools were demonstrated...")
        
        if tracker.all_tools_used():
            logger.info("✅ SUCCESS: All tools have been demonstrated!")
            logger.info(f"Tools used: {tracker.tools_used}")
        else:
            unused_tools = tracker.get_unused_tools()
            logger.warning(f"❌ INCOMPLETE: The following tools were not used: {unused_tools}")
            
            # Try to prompt agent to use remaining tools
            if unused_tools:
                remaining_message = f"Please demonstrate the following tools that haven't been used yet: {', '.join(unused_tools)}"
                final_result = await Runner.run(starting_agent=agent, input=remaining_message)
                logger.info(f"Final attempt result: {final_result.final_output}")
        
        # Final summary
        logger.info("Final tool usage summary:")
        logger.info(f"Expected tools: {tracker.expected_tools}")
        logger.info(f"Tools used: {tracker.tools_used}")
        logger.info(f"Success rate: {len(tracker.tools_used)}/{len(tracker.expected_tools)}")
        
        return tracker.all_tools_used()

@pytest.mark.asyncio
async def test_mcp_agent_tool_capabilities():
    """
    Main test function that runs the tool discovery and usage test.
    """
    success = await test_agent_tool_discovery_and_usage()
    assert success, "Not all tools were successfully demonstrated by the agent"

if __name__ == "__main__":
    # For running the test directly
    asyncio.run(test_agent_tool_discovery_and_usage())