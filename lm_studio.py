import requests
from langchain.llms.base import LLM
from pydantic import Field
from langsmith import Client as traceable
from dotenv import load_dotenv

load_dotenv()

class LMStudioLLM(LLM):
    path: str = Field(...)
    endpoint: str = None

    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.endpoint = f"http://localhost:1234/v1/{self.path}"

    # @traceable
    def _call(self, prompt: str, stop: list = None) -> str:
        response = requests.post(self.endpoint, 
                                json={
                                     "prompt": prompt, 
                                     "model": 
                                    #  "meta-llama-3.1-8b-instruct"
                                     "llama-3.2-3b-instruct"
                                     })
        response_json = response.json()
        return response_json['choices'][0]['text']

    @property
    def _llm_type(self):
        return "lm_studio"