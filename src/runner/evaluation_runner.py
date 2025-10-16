import json
import os
import logging
import yaml
import asyncio
from pathlib import Path
from typing import List, Dict, Any

from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerSse

from runner.base import RunnerBase
from infra.llm import LLMFactory
from models.configs.eval import EvaluationConfig, AgentTest, HumanTest, LLMTest, MCPStdioConfig, MCPSseConfig
from utils.config_manager import ConfigManager



class TestLoadError(Exception):
    """Custom exception for test loading errors."""
    pass

class EvaluationRunner(RunnerBase):
    
    def __init__(self):
        super().__init__()
        config_manager = ConfigManager()
        
        self.full_config = config_manager.config
        self.config: EvaluationConfig  = config_manager.config.eval
        
        self.output_dir = f"output/{self.full_config.run_id}/"
        self.llm_provider = LLMFactory().create(self.config.llm) 
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_all_tests(self) -> List[LLMTest | HumanTest | AgentTest]:
        json_file_path: str = self.config.test.load_test
        yaml_tests: List[LLMTest | HumanTest | AgentTest] = self.config.test.tests

        all_tests = []

        # Add tests from YAML config
        all_tests.extend(yaml_tests)

        # Load tests from JSON file if path is provided
        if json_file_path:
            try:
                json_path = Path(json_file_path)
                if not json_path.exists():
                    raise TestLoadError(f"JSON test file not found: {json_file_path}")

                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                # Try to get tests from 'test_cases' key first, then any key, then error
                json_tests = None
                if 'test_cases' in json_data:
                    json_tests = json_data['test_cases']
                elif 'tests' in json_data:
                    json_tests = json_data['tests']
                else:
                    # Try to find any key that contains a list
                    for key, value in json_data.items():
                        if isinstance(value, list) and value:
                            json_tests = value
                            logging.warning(f"Using tests from key '{key}' in {json_file_path}")
                            break

                if json_tests is None:
                    raise TestLoadError(f"No valid test data found in {json_file_path}. Expected 'test_cases', 'tests', or any list key.")

                # Convert JSON test data to test objects
                for test_data in json_tests:
                    try:
                        test_type = test_data.get('type')
                        if test_type == 'agent':
                            all_tests.append(AgentTest(**test_data))
                        elif test_type == 'llm':
                            all_tests.append(LLMTest(**test_data))
                        elif test_type == 'human':
                            all_tests.append(HumanTest(**test_data))
                        else:
                            logging.error(f"Unknown test type '{test_type}' in test: {test_data.get('name', 'unnamed')}")
                    except Exception as e:
                        logging.error(f"Failed to parse test case {test_data.get('name', 'unnamed')}: {e}")

            except FileNotFoundError:
                raise TestLoadError(f"JSON test file not found: {json_file_path}")
            except json.JSONDecodeError as e:
                raise TestLoadError(f"Invalid JSON in test file {json_file_path}: {e}")
            except Exception as e:
                logging.error(f"Error loading tests from {json_file_path}: {e}")
                raise TestLoadError(f"Failed to load tests from {json_file_path}: {e}")

        if not all_tests:
            logging.warning("No tests loaded from any source")
        else:
            logging.info(f"Loaded {len(all_tests)} tests total")

        return all_tests
    

    async def run(
            self, 
        ):

        tests: List[LLMTest | HumanTest | AgentTest] = self._load_all_tests

        for test in tests:
            
            # handle each case
            pass

        await self._generate_report()


    async def _llm_test(self, test: LLMTest):
        pass

    async def _human_test(self, test: HumanTest):
        pass


    async def _agent_test(self, test: AgentTest):
        """
        Execute an agent test with MCP server integration.

        Args:
            test: AgentTest configuration containing query, MCP config, and prompt

        Returns:
            Dict containing test results with response and execution metadata
        """
        logging.info(f"Running agent test: {test.name}")

        # Create MCP server based on config type
        mcp_server = None
        try:
            if isinstance(test.mcp, MCPStdioConfig):
                # Local stdio MCP server
                logging.info(f"Creating stdio MCP server: {test.mcp.command} {' '.join(test.mcp.args or [])}")
                mcp_server = MCPServerStdio(
                    name=test.name,
                    params={
                        "command": test.mcp.command,
                        "args": test.mcp.args or [],
                        "env": test.mcp.env or {},
                        **({"cwd": test.mcp.cwd} if test.mcp.cwd else {})
                    }
                )
            elif isinstance(test.mcp, MCPSseConfig):
                # Remote SSE MCP server
                logging.info(f"Creating SSE MCP server: {test.mcp.url}")
                mcp_server = MCPServerSse(
                    name=test.name,
                    params={
                        "url": test.mcp.url,
                        "headers": test.mcp.headers or {},
                        "timeout": test.mcp.timeout or 30.0,
                        "sse_read_timeout": test.mcp.sse_read_timeout or 300.0
                    }
                )
            else:
                raise ValueError(f"Unknown MCP config type: {type(test.mcp)}")

            # Initialize the agent with MCP server
            async with mcp_server:
                # Determine model to use (test-specific or global)
                model = test.agent_model or self.config.llm.model if self.config.llm else "gpt-4o-mini"

                # Create agent with custom instructions
                agent_instructions = test.agent_instructions or """You are a helpful AI assistant with access to external tools.
Your task is to answer the user's query using the provided context and available tools.
Be precise and use tools when necessary to provide accurate information."""

                agent = Agent(
                    name=test.name,
                    model=model,
                    instructions=agent_instructions,
                    mcp_servers=[mcp_server]
                )

                # Prepare the prompt with retrieved context
                # TODO: Retrieve relevant documents based on test.query
                # For now, use the query and prompt directly
                user_message = test.prompt or test.query

                logging.info(f"Executing agent with message: {user_message[:100]}...")

                # Execute the agent with timeout
                runner = Runner()
                try:
                    result = await asyncio.wait_for(
                        runner.run(agent=agent, user_message=user_message),
                        timeout=test.timeout or 60
                    )

                    # Extract response and tool calls
                    response_text = ""
                    tools_called = []

                    # Process the result (structure depends on agent response format)
                    if hasattr(result, 'messages'):
                        for msg in result.messages:
                            if hasattr(msg, 'content'):
                                response_text += str(msg.content)
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    if hasattr(tool_call, 'function'):
                                        tools_called.append(tool_call.function.name)
                    elif isinstance(result, str):
                        response_text = result
                    else:
                        response_text = str(result)

                    # Compile test results
                    test_result = {
                        "test_name": test.name,
                        "status": "completed",
                        "query": test.query,
                        "prompt": user_message,
                        "response": response_text,
                        "tools_called": tools_called,
                        "model": model,
                        "execution_time": None  # TODO: Add timing
                    }

                    logging.info(f"Agent test {test.name} completed successfully")
                    return test_result

                except asyncio.TimeoutError:
                    logging.error(f"Agent test {test.name} timed out after {test.timeout}s")
                    return {
                        "test_name": test.name,
                        "status": "timeout",
                        "error": f"Test timed out after {test.timeout} seconds"
                    }

        except Exception as e:
            logging.error(f"Error running agent test {test.name}: {e}", exc_info=True)
            return {
                "test_name": test.name,
                "status": "error",
                "error": str(e)
            }


    async def _generate_report(self):
        """Generate YAML and Markdown reports of the current configuration."""

        # Convert config to dict for YAML serialization
        config_dict = self.full_config.model_dump()

        # Save YAML version
        yaml_path = os.path.join(self.output_dir, "config.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        # Generate and save Markdown version (exclude tests)
        md_config = config_dict.copy()
        if 'eval' in md_config and 'test' in md_config['eval'] and 'tests' in md_config['eval']['test']:
            md_config['eval']['test']['tests'] = f"[{len(md_config['eval']['test']['tests'])} test cases - see YAML for details]"

        md_path = os.path.join(self.output_dir, "config.md")
        with open(md_path, 'w') as f:
            f.write(self._config_to_markdown(md_config))

        logging.info(f"Generated config reports: {yaml_path}, {md_path}")

    def _config_to_markdown(self, config: Dict[str, Any]) -> str:
        """Convert configuration dictionary to human-readable Markdown."""
        md_lines = ["# Configuration Report\n"]

        for section_name, section_data in config.items():
            if section_data is None:
                continue

            md_lines.append(f"## {section_name.replace('_', ' ').title()}\n")

            if isinstance(section_data, dict):
                md_lines.extend(self._dict_to_markdown_table(section_data))
            elif isinstance(section_data, list):
                md_lines.append("### Items:")
                for i, item in enumerate(section_data, 1):
                    if isinstance(item, dict):
                        md_lines.append(f"\n**{i}.** {item.get('name', f'Item {i}')}")
                        md_lines.extend(self._dict_to_markdown_table(item, level=3))
                    else:
                        md_lines.append(f"- {item}")
            else:
                md_lines.append(f"**Value:** `{section_data}`\n")

            md_lines.append("")

        return "\n".join(md_lines)

    def _dict_to_markdown_table(self, data: Dict[str, Any], level: int = 2) -> List[str]:
        """Convert dictionary to markdown table format."""
        if not data:
            return ["*No configuration data*\n"]

        lines = []

        # Simple key-value pairs
        simple_pairs = []
        complex_items = {}

        for key, value in data.items():
            if isinstance(value, (dict, list)) and value:
                complex_items[key] = value
            else:
                simple_pairs.append((key, value))

        # Add simple key-value table
        if simple_pairs:
            lines.extend([
                "| Setting | Value |",
                "|---------|-------|"
            ])

            for key, value in simple_pairs:
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, bool):
                    formatted_value = "✅ Yes" if value else "❌ No"
                elif isinstance(value, list):
                    formatted_value = ", ".join(str(v) for v in value)
                elif value is None:
                    formatted_value = "*Not set*"
                else:
                    formatted_value = f"`{value}`"

                lines.append(f"| {formatted_key} | {formatted_value} |")

            lines.append("")

        # Add complex nested items
        for key, value in complex_items.items():
            header_level = "#" * (level + 1)
            lines.append(f"{header_level} {key.replace('_', ' ').title()}")

            if isinstance(value, dict):
                lines.extend(self._dict_to_markdown_table(value, level + 1))
            elif isinstance(value, list):
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        lines.append(f"\n**{i}.** {item.get('name', item.get('type', f'Item {i}'))}")
                        lines.extend(self._dict_to_markdown_table(item, level + 2))
                    else:
                        lines.append(f"- {item}")

            lines.append("")

        return lines