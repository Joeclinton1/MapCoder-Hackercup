from mapcoder_hackercup.models.Gemini import Gemini
from mapcoder_hackercup.models.OpenAI import ChatGPT
from mapcoder_hackercup.models.OpenAI import GPT4
from mapcoder_hackercup.models.Ollama import Codestral
from mapcoder_hackercup.models.Ollama import Local

class ModelFactory:
    @staticmethod
    def get_model_class(model_name):
        if model_name == "Gemini":
            return Gemini
        elif model_name == "ChatGPT":
            return ChatGPT
        elif model_name == "GPT4":
            return GPT4
        elif model_name == "Codestral":
            return Codestral
        elif model_name == "Local":
            return Local
        else:
            raise Exception(f"Unknown model name {model_name}")
