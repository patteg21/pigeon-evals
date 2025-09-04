from typing import Optional, Dict, List, Union
import json

from openai import AsyncOpenAI

from utils import logger



class OpenAILLM:

    def __init__(self):
        self.client = AsyncOpenAI()
        pass

    def generate_response(
            self,
            query: str,
            model: str ='gpt-4o-mini',
            history: List[Dict] = [],
            prompt: str = "",
            mode: Optional[Dict] = { "type": "json_object" },
            max_tokens: int = 8192,
        ) -> Union[str, Dict]:
        """
        Makes a call to Azure OpenAI Services
        
        :param query: the main query to be answered or asked
        :param prompt: the prompt for the llm, going to system
        :param mode: what type of response will the ai output
        :param additional_context: a list of the history of the chat or system response

        :return response: a string response from openai llm
        """
        try:
            messages=[
                    {'role': 'system', 'content': prompt},
                    *history,
                    {'role': 'user', 'content': [
                            {
                            'type':"text", "text": query ,
                            }
                        ]
                    }
                ]

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=mode,
                max_tokens=max_tokens
            )
            res = response.choices[0].message.content

            if mode == { "type": "json_object" }:
                return json.loads(res)

            return res
        except Exception as e:
            logger.error(f"Azure OpenAI Error: {e}")
            res = f"This is a sample Generated Response: {e}"