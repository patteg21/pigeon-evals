# Agent Testing with MCP Servers

This guide explains how to implement and run agent tests using the Model Context Protocol (MCP) servers, supporting both local (stdio) and remote (SSE) configurations.

## Overview

Agent tests allow you to evaluate AI agents that have access to external tools through MCP servers. These tests execute agents and capture their outputs for review. The implementation supports:

- **Local MCP servers** using stdio (similar to Claude Code's format)
- **Remote MCP servers** using Server-Sent Events (SSE)
- Configurable timeouts and turn limits
- Custom agent instructions and models
- Output capture with tool usage tracking

## Configuration Structure

### Local MCP Server (stdio)

```yaml
- type: "agent"
  name: "calculator_agent_test"
  query: "What is 15 multiplied by 23?"
  prompt: "Calculate 15 * 23 and explain the result"
  mcp:
    type: "stdio"
    command: "python"
    args:
      - "-m"
      - "calculator_mcp_server"
    env:
      DEBUG: "true"
    cwd: "./mcp_servers"
  timeout: 30
  max_turns: 5
  agent_model: "gpt-4o-mini"
```

### Remote MCP Server (SSE)

```yaml
- type: "agent"
  name: "weather_agent_test"
  query: "What's the weather in San Francisco?"
  prompt: "Get the current weather for San Francisco and summarize it"
  mcp:
    type: "sse"
    url: "https://api.example.com/mcp/weather"
    headers:
      Authorization: "Bearer ${WEATHER_API_KEY}"
    timeout: 30.0
    sse_read_timeout: 300.0
  timeout: 45
  agent_instructions: "You are a weather assistant. Use the weather tool to get accurate real-time data."
```

## Field Reference

### Required Fields

- `type`: Must be "agent"
- `name`: Unique identifier for the test
- `query`: Query used for retrieval from vector database
- `mcp`: MCP server configuration (see below)

### Optional Fields

#### Prompts and Instructions
- `prompt`: The message sent to the agent (defaults to `query`)
- `agent_instructions`: Custom system instructions for the agent

#### Execution Configuration
- `timeout`: Maximum execution time in seconds (default: 60)
- `max_turns`: Maximum conversation turns (default: 10)
- `agent_model`: Model to use for this test (overrides global config)

### MCP Configuration

#### stdio Type (Local)
```yaml
mcp:
  type: "stdio"
  command: "python"              # Required: command to run
  args: ["-m", "server"]         # Optional: command arguments
  env:                           # Optional: environment variables
    DEBUG: "true"
  cwd: "./path"                  # Optional: working directory
```

#### sse Type (Remote)
```yaml
mcp:
  type: "sse"
  url: "https://api.example.com"  # Required: server URL
  headers:                        # Optional: HTTP headers
    Authorization: "Bearer token"
  timeout: 30.0                   # Optional: request timeout (default: 30s)
  sse_read_timeout: 300.0         # Optional: SSE read timeout (default: 300s)
```

## Examples

### Example 1: File System Operations

Test an agent that can list and read files:

```yaml
- type: "agent"
  name: "file_operations_test"
  query: "List files in the data directory"
  prompt: "List all files in ./data and count them"
  mcp:
    type: "stdio"
    command: "npx"
    args:
      - "-y"
      - "@modelcontextprotocol/server-filesystem"
      - "./data"
  timeout: 20
```

### Example 2: Database Queries

Test an agent with SQLite database access:

```json
{
  "type": "agent",
  "name": "sqlite_database_query",
  "query": "Show me all users",
  "prompt": "Query the database to list all users",
  "mcp": {
    "type": "stdio",
    "command": "npx",
    "args": [
      "-y",
      "@modelcontextprotocol/server-sqlite",
      "--db-path",
      "./data/users.db"
    ]
  },
  "timeout": 30
}
```

### Example 3: Web Search

Test an agent with Brave Search capabilities:

```json
{
  "type": "agent",
  "name": "brave_search_test",
  "query": "Search for recent AI developments",
  "prompt": "Search for information about recent AI model developments in 2025",
  "mcp": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
    "env": {
      "BRAVE_API_KEY": "${BRAVE_API_KEY}"
    }
  },
  "timeout": 45,
  "max_turns": 8
}
```

## Environment Variables

You can use environment variable substitution in your configuration:

```yaml
mcp:
  type: "sse"
  url: "${MCP_SERVER_URL}"
  headers:
    Authorization: "Bearer ${API_TOKEN}"
```

Set the variables before running tests:
```bash
export API_TOKEN="your-token-here"
export MCP_SERVER_URL="https://api.example.com/mcp"
```

## Running Tests

### Via YAML Configuration

Add your agent tests to `configs/eval.yml`:

```yaml
eval:
  test:
    tests:
      - type: "agent"
        name: "my_agent_test"
        # ... configuration
```

### Via JSON File

Create a JSON file with your tests and reference it:

```yaml
eval:
  test:
    load_test: "data/tests/my_agent_tests.json"
```

Then run the evaluation:

```bash
python -m runner.evaluation_runner
```

## Test Results

Each agent test returns:

```json
{
  "test_name": "calculator_agent_test",
  "status": "completed",
  "query": "What is 15 multiplied by 23?",
  "prompt": "Calculate 15 * 23 and explain the result",
  "response": "The result of 15 * 23 is 345.",
  "tools_called": ["calculate"],
  "model": "gpt-4o-mini",
  "execution_time": 2.5
}
```

The test output includes:
- **test_name**: Name of the test
- **status**: `completed`, `timeout`, or `error`
- **query**: The retrieval query used
- **prompt**: The actual message sent to the agent
- **response**: The agent's output
- **tools_called**: List of MCP tools the agent used
- **model**: The model used for this test
- **execution_time**: Time taken (when implemented)

## Common MCP Servers

Here are some popular MCP servers you can use:

- **Filesystem**: `@modelcontextprotocol/server-filesystem`
- **SQLite**: `@modelcontextprotocol/server-sqlite`
- **PostgreSQL**: `@modelcontextprotocol/server-postgres`
- **Brave Search**: `@modelcontextprotocol/server-brave-search`
- **GitHub**: `@modelcontextprotocol/server-github`
- **Google Drive**: `@modelcontextprotocol/server-gdrive`
- **Slack**: `@modelcontextprotocol/server-slack`

## Troubleshooting

### Timeout Errors

If tests timeout, try:
- Increasing the `timeout` value
- Reducing `max_turns`
- Checking if the MCP server is responsive

### Connection Errors (SSE)

For remote MCP servers:
- Verify the URL is correct
- Check authentication headers
- Ensure network connectivity
- Increase `timeout` or `sse_read_timeout`

### MCP Server Not Starting

For stdio MCP servers:
- Verify the command and args are correct
- Check that the working directory (`cwd`) exists
- Ensure environment variables are set properly
- Test the command manually in your terminal

## Best Practices

1. **Start Simple**: Begin with basic tests with minimal configuration
2. **Use Specific Prompts**: Clear prompts lead to more predictable agent behavior
3. **Set Reasonable Timeouts**: Account for network latency and tool execution time
4. **Environment Variables**: Use env vars for sensitive data like API keys
5. **Test Isolation**: Each test should be independent and not rely on previous tests
6. **Review Outputs**: Examine the `tools_called` and `response` fields to understand agent behavior

## References

- [OpenAI Agents Python SDK](http://openai.github.io/openai-agents-python/mcp/)
- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Claude Code MCP Configuration](https://docs.claude.com/)
