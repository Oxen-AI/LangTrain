
from langtrain.models.llms.llm import LLM
import requests
import os
import json

class OpenAICompletion(LLM):
    def __init__(self, model_name: str = "text-davinci-003"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name

    def __call__(
        self,
        prompt: str,
    ) -> str:
        if self.api_key is None:
            raise ValueError("OpenAI API key not set")
        
        url = "https://api.openai.com/v1/completions"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 256
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload))
        response = r.json()
        print(response)
        return response['choices'][0]['text']

        