import os
import dotenv
import requests

from .Base import BaseModel
from mapcoder_hackercup.utils.token_count import token_count

dotenv.load_dotenv()

class OllamaBaseModel(BaseModel):
    """
    Ollama Model interface.

    Arguments
    ---------
    api_url : str
        URL where the Ollama API is hosted. If not provided, defaults to 'http://localhost:11434'.
    model_name : str
        Name of the model to use. If not provided, the implementation will derive it from
        the environment variable `OLLAMA_MODEL`.
    temperature : float
        Temperature value to use for the model. Defaults to zero for reproducibility.
    top_p : float
        Top P value to use for the model. Defaults to 0.95.
    max_tokens : int
        Maximum number of tokens to generate. Defaults to 800.
    frequency_penalty : float
        Frequency Penalty to use for the model.
    presence_penalty : float
        Presence Penalty to use for the model.
    """

    def __init__(
        self,
        api_url=None,
        model_name=None,
        temperature=0,
        top_p=0.95,
        max_tokens=32768,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        self.api_url = api_url or os.getenv("OLLAMA_API_URL") or "http://windows-6absj2b:11434"
        self.model_name = model_name or os.getenv("OLLAMA_MODEL")
        assert self.model_name is not None, "Model name must be provided as model config or environment variable `OLLAMA_MODEL`"

        # Model parameters
        self.model_params = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "num_ctx": 3072
        }

    def prompt(self, processed_input: list[dict], **kwargs):
        """
        Ollama API implementation.

        Arguments
        ---------
        processed_input : list
            Must be a list of dictionaries, where each dictionary has two keys:
            "role" defines a role in the chat (e.g., "system", "user") and
            "content" defines the actual message for that turn.

        Returns
        -------
        response_content : str
            The generated response content.
        prompt_tokens : int
            Number of prompt tokens used.
        completion_tokens : int
            Number of completion tokens generated.
        """
        # Prepare the prompt by concatenating the messages
        prompt = ""
        for message in processed_input:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"{content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        # Build the payload
        self.model_params.update(kwargs)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options":self.model_params,
            "stream": False
        }

        # Send the request to the Ollama API
        response = requests.post(f"{self.api_url}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract the response content
        response_content = data.get("response", "")
        # with open("./debug.txt", "w") as f:
        #     # logging
        #     f.write(response_content)
        # Extract token counts
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return response_content, prompt_tokens, completion_tokens

# Specific model classes can be defined if needed
class Codestral(OllamaBaseModel):
    def __init__(self, *args, **kwargs):
        kwargs['model_name'] = 'codestral:latest'
        super().__init__(*args, **kwargs)

class Local(OllamaBaseModel):
    def __init__(self, *args, **kwargs):
        # Fetch the first local model
        api_url = os.getenv("OLLAMA_API_URL") or "http://windows-6absj2b:11434"
        first_model = requests.get(f"{api_url}/api/tags").json()["models"][0]["model"]
        kwargs['model_name'] = first_model
        super().__init__(*args, **kwargs)

class Deepseek(OllamaBaseModel):
    def __init__(self, *args, **kwargs):
        kwargs['model_name'] = 'deepseek-coder-v2:lite '
        super().__init__(*args, **kwargs)


class Llama(OllamaBaseModel):
    def __init__(self, *args, **kwargs):
        # Fetch the required API URL from the environment variable, raise exception if not set
        api_url = os.getenv("OLLAMA_API_URL_A100")
        if not api_url:
            raise EnvironmentError("Environment variable 'OLLAMA_API_URL_A100' must be set.")

        kwargs['model_name'] = 'llama3.1:70b-instruct-q3_K_M'
        super().__init__(api_url=api_url, *args, **kwargs)
