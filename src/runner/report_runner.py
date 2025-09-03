from typing import List
from uuid import uuid4
import json
import os

from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from agents import Agent, Runner, set_default_openai_key

from src.embedder.openai_embedder import OpenAIEmbedder, BaseEmbedder
from src.storage.vector import PineconeDB, VectorStorageBase
from src.storage.text import SQLiteDB, TextStorageBase 
from src.storage.vector.reranker import HuggingFaceReranker

from src.utils.types import LLMTest, AgentTest, HumanTest, ReportConfig, YamlConfig


load_dotenv()


class ReportRunner:

    def __init__(self):
        
        self.provider_map = {
            "huggingface" : HuggingFaceReranker
        }

    async def run_report(
            self, 
            report_config: ReportConfig, 
            config: YamlConfig,
            *, 
            vector_db_client: BaseEmbedder | None = None,
            embedding_model: VectorStorageBase | None = None,
            sql_client: TextStorageBase | None = None
        ):

        self.vector_db_client = vector_db_client or PineconeDB()
        self.embedding_model = embedding_model or OpenAIEmbedder(pca_path="data/artifacts/pca_512.joblib")
        self.sql_client = sql_client or SQLiteDB()

        # adds in a reranker for search when relevant (Not in MCP Test)
        self.reranker = None
        if report_config.retrieval and report_config.retrieval.rerank:
            self.reranker = HuggingFaceReranker(report_config.retrieval.rerank)
            
        if report_config.retrieval:
            self.top_k = report_config.retrieval.top_k



        self.evaluations = report_config.evaluations or True
        self.metrics: List = report_config.metrics


        default_tests: List = self._read_test_cases()

        self.run_id = config.run_id
        self.output_path = report_config.output_path if report_config.output_path else f"evals/reports/{self.run_id}/"
        self.original_config = config


        tests =  report_config.tests + default_tests

        for test in tests:
            if test.type == "agent":
                await self._execute_agent_test(test)
            elif test.type == "human":
                await self._execute_human_test(test)
            elif test.type == "llm":
                await self._execute_llm_test(test)
        
        self._write_yaml()
    
    

    def _read_test_cases(self, path: str = "data/tests/default.json") -> List[AgentTest | HumanTest | LLMTest]:
        # TODO match * to search for all formatted json in correct format log when incorrect format
        
        return []


    async def _execute_agent_test(self, agent_test: AgentTest):
        openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not set...")
        set_default_openai_key(openai_api_key)

        agent_test.mcp.args

        params = {
            "command": agent_test.mcp.command,
            "args": agent_test.mcp.args
        }

        async with MCPServerStdio(params=params) as server:
            agent = Agent(
                name="Assistant",
                instructions=f"{agent_test.prompt}",
                mcp_servers=[server],
            )
            
            result = await Runner.run(starting_agent=agent, input=agent_test.query)

        
        self._write_outputs(
            agent_test.model_dump(), 
            output=result.final_output, 
            name=agent_test.name or f"agent_test_{uuid4().hex}"
        )


    async def _execute_human_test(self, human_test: HumanTest):

        top_k = self.top_k or 10

        vec = await self.embedding_model.create_pinecone_embeddings(human_test.query)


        response = self.vector_db_client.query(vec, top_k=top_k, include_metadata=True)

        similiarity_search: List = []
        for match in response.matches:
            metadata = getattr(match, 'metadata', {}) or {}
                    
            # Replace any existing text with full text from SQL database using document_id
            document_id = metadata.get('document_id')
            if document_id:

                doc = self.sql_client.retrieve_document(document_id)
                if doc and doc.get('text'):
                    metadata['text'] = doc['text']  # Replace with full text from SQL
            similiarity_search.append(metadata)

        if self.reranker:
            similiarity_search = self.reranker.rerank(documents=similiarity_search, query=human_test.query)

        self._write_outputs(
            human_test.model_dump(),
            similiarity_search,
            human_test.name or f"human_test_{uuid4().hex}"
        )
        
    async def _execute_llm_test(self, llm_test: LLMTest):

        
        top_k = self.top_k or 10

        vec = await self.embedding_model.create_pinecone_embeddings(llm_test.query)


        response = self.vector_db_client.query(vec, top_k=top_k, include_metadata=True)

        similiarity_search: List = []
        for match in response.matches:
            metadata = getattr(match, 'metadata', {}) or {}
                    
                    # Replace any existing text with full text from SQL database using document_id
            document_id = metadata.get('document_id')
            if document_id:

                doc = self.sql_client.retrieve_document(document_id)
                if doc and doc.get('text'):
                    metadata['text'] = doc['text']  # Replace with full text from SQL
            similiarity_search.append(metadata)
        
        if self.reranker:
            similiarity_search = self.reranker.rerank(documents=similiarity_search, query=llm_test.query)
        
        prompt = llm_test.prompt if llm_test.prompt else " " # TODO: Add in a default prompt 
        
        agent = Agent(
            name="Assistant",
            instructions=prompt,
        )
        
        result = await Runner.run(starting_agent=agent, input=llm_test.query)



        self._write_outputs(
            llm_test.model_dump(),
            {
                "judge": result.final_output,
                "search": similiarity_search,
            },
            llm_test.name or f"llm_test{uuid4().hex}"
        )

    
    def _write_outputs(self, input, output, name):
        os.makedirs(self.output_path, exist_ok=True)
        
        output_data = {
            "input": input,
            "output": output,
            "name": name
        }
    
        filename = f"{name}.json"
        filepath = os.path.join(self.output_path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def _write_yaml(self):
        os.makedirs(self.output_path, exist_ok=True)
        
        config_filename = f"{self.original_config.task}_config.json"
        config_filepath = os.path.join(self.output_path, config_filename)
        
        with open(config_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.original_config.model_dump(), f, indent=2, ensure_ascii=False)
        