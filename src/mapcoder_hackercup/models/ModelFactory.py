from mapcoder_hackercup.models.Gemini import Gemini
from mapcoder_hackercup.models.OpenAI import ChatGPT, GPT4
from mapcoder_hackercup.models.OpenAI import CodestralVLLM, LlamaVLLM, QwenVLLM
from mapcoder_hackercup.models.Ollama import Local, Llama, Deepseek, Codestral
from mapcoder_hackercup.models.Claude import Claude35Sonnet
from mapcoder_hackercup.models.Base import BaseModel
from typing import Type

class ModelFactory:
    @staticmethod
    def get_model_class(model_name) -> Type[BaseModel]:
        match model_name:
            case "Gemini":
                return Gemini
            case "ChatGPT":
                return ChatGPT
            case "GPT4":
                return GPT4
            case "Codestral":
                return Codestral
            case "Local":
                return Local
            case "Llama":
                return Llama
            case "Deepseek":
                return Deepseek
            case "CodestralVLLM":
                return CodestralVLLM
            case "LlamaVLLM":
                return LlamaVLLM
            case "QwenVLLM":
                return QwenVLLM
            case "Claude35Sonnet":
                return Claude35Sonnet
            case _:
                raise Exception(f"Unknown model name {model_name}")
