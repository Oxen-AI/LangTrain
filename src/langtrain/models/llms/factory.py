
from langtrain.models.llms.llm import LLM
from langtrain.models.llms.flan_t5 import FlanT5
from langtrain.models.llms.cerebras import CerebrasLLM
from langtrain.models.llms.openai import OpenAICompletion

def get_model(name: str) -> LLM:
    if name == "hf/google/flan-t5-small":
        return FlanT5('google/flan-t5-small')
    elif name == "hf/google/flan-t5-large":
        return FlanT5('google/flan-t5-large')
    elif name == "hf/google/flan-t5-xl":
        return FlanT5('google/flan-t5-xl')
    elif name == "hf/cerebras/Cerebras-GPT-2.7B":
        return CerebrasLLM('cerebras/Cerebras-GPT-2.7B')
    elif name == "hf/cerebras/Cerebras-GPT-111M-cpu":
        return CerebrasLLM('cerebras/Cerebras-GPT-111M', device='cpu')
    elif name == "api/openai/text-davinci-003":
        return OpenAICompletion(model_name="text-davinci-003")
    else:
        raise ValueError(f"Unknown model {name}")
