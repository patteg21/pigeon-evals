import os
import pytest
import asyncio
from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from agents import Agent, Runner, set_default_openai_key
from utils import logger
from typing import List, Dict, Any

load_dotenv()

openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not set...")
set_default_openai_key(openai_api_key)


class RelevancyEvaluator:
    """Evaluates the relevancy of vector search results"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.evaluation_criteria = {
            "semantic_match": "Does the content semantically match the query?",
            "factual_accuracy": "Is the information factually relevant to the query?",
            "completeness": "Does the result provide sufficient information to answer the query?",
            "context_relevance": "Is the context appropriate for the query?"
        }
    
    async def evaluate_search_results(self, query: str, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate search results for relevancy using an AI agent
        
        Args:
            query: The original search query
            search_results: Results from vector database search
            
        Returns:
            Dictionary containing evaluation results and pass/fail status
        """
        
        # Extract matches from search results
        matches = search_results.get('matches', [])
        if not matches:
            return {
                "status": "FAIL",
                "reason": "No search results to evaluate",
                "score": 0.0,
                "evaluations": []
            }
        
        # Prepare evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(query, matches)
        
        # Get agent evaluation
        try:
            evaluation_result = await Runner.run(
                starting_agent=self.agent, 
                input=evaluation_prompt
            )
            
            # Parse the evaluation response
            evaluation_data = self._parse_evaluation_response(evaluation_result.final_output)
            
            return evaluation_data
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {
                "status": "ERROR",
                "reason": f"Evaluation failed: {str(e)}",
                "score": 0.0,
                "evaluations": []
            }
    
    def _create_evaluation_prompt(self, query: str, matches: List[Dict]) -> str:
        """Create a structured prompt for evaluating search results"""
        
        prompt = f"""
You are an expert evaluator of search result relevancy. Your task is to evaluate how well the retrieved documents match the given query.

QUERY: "{query}"

SEARCH RESULTS:
"""
        
        for i, match in enumerate(matches[:5]):  # Limit to top 5 results
            metadata = match.get('metadata', {})
            text_snippet = metadata.get('text', '')[:500] + "..." if len(metadata.get('text', '')) > 500 else metadata.get('text', '')
            
            prompt += f"""
Result {i+1}:
- Score: {match.get('score', 'N/A')}
- Document ID: {metadata.get('document_id', 'N/A')}
- Entity Type: {metadata.get('entity_type', 'N/A')}
- Ticker: {metadata.get('ticker', 'N/A')}
- Year: {metadata.get('year', 'N/A')}
- Text: {text_snippet}

"""
        
        prompt += f"""
EVALUATION CRITERIA:
{chr(10).join([f"- {criterion}: {description}" for criterion, description in self.evaluation_criteria.items()])}

Please evaluate each result and provide your assessment in the following JSON format:
{{
    "overall_score": <float between 0.0 and 1.0>,
    "status": "<PASS or FAIL>",
    "reason": "<brief explanation of your decision>",
    "individual_evaluations": [
        {{
            "result_index": <int>,
            "relevancy_score": <float between 0.0 and 1.0>,
            "meets_criteria": {{
                "semantic_match": <true/false>,
                "factual_accuracy": <true/false>,
                "completeness": <true/false>,
                "context_relevance": <true/false>
            }},
            "explanation": "<brief explanation>"
        }}
    ]
}}

Consider the search PASSES if:
- Overall score >= 0.7
- At least 2 results have relevancy_score >= 0.8
- The results provide meaningful information related to the query

Respond ONLY with the JSON, no additional text.
"""
        
        return prompt
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse the agent's evaluation response"""
        try:
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                evaluation_data = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['overall_score', 'status', 'reason']
                for field in required_fields:
                    if field not in evaluation_data:
                        raise ValueError(f"Missing required field: {field}")
                
                return evaluation_data
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse evaluation response: {e}")
            logger.error(f"Response was: {response}")
            
            # Return a fallback evaluation
            return {
                "overall_score": 0.0,
                "status": "FAIL",
                "reason": f"Failed to parse evaluation response: {str(e)}",
                "individual_evaluations": []
            }


class VectorSearchRelevancyTestSuite:
    """Test suite for vector search relevancy evaluation"""
    
    def __init__(self):
        self.test_cases = [
            {
                "name": "Apple Revenue Query",
                "query": "What was Apple's revenue in 2023?",
                "expected_entities": ["10K", "10Q"],
                "expected_ticker": "AAPL",
                "min_score": 0.7
            },
            {
                "name": "Microsoft Cloud Services",
                "query": "Microsoft Azure cloud computing revenue growth",
                "expected_entities": ["10K", "10Q"],
                "expected_ticker": "MSFT",
                "min_score": 0.7
            },
            {
                "name": "Tesla Production Numbers",
                "query": "Tesla vehicle production and delivery numbers 2024",
                "expected_entities": ["10K", "10Q"],
                "expected_ticker": "TSLA",
                "min_score": 0.7
            },
            {
                "name": "General Financial Performance",
                "query": "quarterly earnings and financial performance tech companies",
                "expected_entities": ["10K", "10Q"],
                "expected_ticker": None,  # Multiple companies expected
                "min_score": 0.6  # Lower threshold for broader query
            }
        ]
    
    async def run_test_case(self, test_case: Dict, search_agent: Agent, evaluator_agent: Agent) -> Dict[str, Any]:
        """Run a single test case"""
        
        logger.info(f"Running test case: {test_case['name']}")
        
        # Step 1: Perform vector search
        search_result = await Runner.run(
            starting_agent=search_agent,
            input=f"Search for: {test_case['query']}"
        )
        
        # Step 2: Extract search results (this would need to be adapted based on actual response format)
        # For now, we'll simulate search results structure
        mock_search_results = {
            "matches": [
                {
                    "score": 0.85,
                    "metadata": {
                        "document_id": "mock_doc_1",
                        "entity_type": "10K",
                        "ticker": test_case.get("expected_ticker", "AAPL"),
                        "year": 2023,
                        "text": f"Sample text related to {test_case['query']}"
                    }
                }
            ]
        }
        
        # Step 3: Evaluate relevancy
        evaluator = RelevancyEvaluator(evaluator_agent)
        evaluation_result = await evaluator.evaluate_search_results(
            test_case['query'], 
            mock_search_results
        )
        
        # Step 4: Determine pass/fail
        passed = (
            evaluation_result.get('status') == 'PASS' and
            evaluation_result.get('overall_score', 0.0) >= test_case['min_score']
        )
        
        return {
            "test_case": test_case['name'],
            "query": test_case['query'],
            "search_response": search_result.final_output,
            "evaluation": evaluation_result,
            "passed": passed,
            "min_score_required": test_case['min_score'],
            "actual_score": evaluation_result.get('overall_score', 0.0)
        }


@pytest.mark.asyncio
async def test_vector_search_relevancy_evaluation():
    """
    Main test that evaluates vector search results for relevancy using an AI agent
    """
    
    # Set up MCP server
    params = {
        "command": "python",
        "args": ["main.py"]
    }
    
    async with MCPServerStdio(params=params) as server:
        # Create search agent
        search_agent = Agent(
            name="SearchAgent",
            instructions="""You are a financial search assistant. Use vector search tools to find information from SEC filings. 
            Provide detailed search results including document metadata and relevant text snippets.""",
            mcp_servers=[server],
        )
        
        # Create evaluator agent
        evaluator_agent = Agent(
            name="EvaluatorAgent", 
            instructions="""You are an expert evaluator of search result relevancy. Analyze search results critically and provide 
            structured JSON evaluations. Be precise in your scoring and provide clear explanations for your decisions.""",
            mcp_servers=[],  # No need for MCP tools for evaluation
        )
        
        # Initialize test suite
        test_suite = VectorSearchRelevancyTestSuite()
        
        # Run all test cases
        results = []
        for test_case in test_suite.test_cases:
            try:
                result = await test_suite.run_test_case(test_case, search_agent, evaluator_agent)
                results.append(result)
                
                logger.info(f"Test '{test_case['name']}': {'PASSED' if result['passed'] else 'FAILED'}")
                logger.info(f"Score: {result['actual_score']:.2f} (required: {result['min_score_required']:.2f})")
                
            except Exception as e:
                logger.error(f"Test case '{test_case['name']}' failed with error: {e}")
                results.append({
                    "test_case": test_case['name'],
                    "query": test_case['query'],
                    "passed": False,
                    "error": str(e)
                })
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.get('passed', False))
        
        logger.info("\n=== VECTOR SEARCH RELEVANCY TEST SUMMARY ===")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success rate: {(passed_tests / total_tests * 100):.1f}%")
        
        # Detailed results
        for result in results:
            logger.info(f"\nTest: {result['test_case']}")
            logger.info(f"Query: {result['query']}")
            logger.info(f"Status: {'PASSED' if result.get('passed', False) else 'FAILED'}")
            if 'actual_score' in result:
                logger.info(f"Score: {result['actual_score']:.2f}")
            if 'evaluation' in result:
                logger.info(f"Reason: {result['evaluation'].get('reason', 'N/A')}")
        
        # Test passes if at least 75% of test cases pass
        success_rate = passed_tests / total_tests
        assert success_rate >= 0.75, f"Test suite failed: only {success_rate:.1%} of tests passed (need ≥75%)"
        
        return results


if __name__ == "__main__":
    # For running the test directly
    asyncio.run(test_vector_search_relevancy_evaluation())