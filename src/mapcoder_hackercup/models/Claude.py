import os
import dotenv
from anthropic import Anthropic

from .Base import BaseModel
from mapcoder_hackercup.utils.token_count import token_count

dotenv.load_dotenv()

class ClaudeBaseModel(BaseModel):
    """
    Claude Model interface. Can be used for models hosted on Anthropic's platform.

    Arguments
    ---------
    api_key : str
        Authentication token for the API. If not provided, the implementation will derive it
        from environment variable `ANTHROPIC_API_KEY`.
    model_name : str
        Name of the model to use. If not provided, the implementation will derive it from
        environment variable `ANTHROPIC_MODEL`
    temperature : float
        Temperature value to use for the model. Defaults to 0.7 for a balance of creativity and consistency.
    top_p : float
        Top P value to use for the model. Defaults to 1 (disabled)
    max_tokens : int
        Maximum number of tokens to pass to the model. Defaults to 1000
    """

    def __init__(
            self,
            api_key=None,
            model_name=None,
            temperature=0.7,
            top_p=1,
            max_tokens=8192,
            gpu=''
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        assert self.api_key is not None, "API Key must be provided as model config or environment variable `ANTHROPIC_API_KEY`"

        self.anthropic = Anthropic(api_key=self.api_key)

        # Claude parameters
        self.model_params = {
            "model": model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

    def summarize_response(self, response):
        """Returns the content of the response"""
        return response.content

    def prompt(self, processed_input: list[dict]):
        """
        Anthropic API ChatCompletion implementation

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "human", "assistant") and
            "content" defines the actual message for that turn

        Returns
        -------
        response : str
            Content of the response from Claude
        prompt_tokens : int
            Number of tokens in the prompt
        completion_tokens : int
            Number of tokens in the completion
        """

        # Convert OpenAI-style roles to Anthropic roles
        role_mapping = {"system": "user", "user": "user", "assistant": "assistant"}
        messages = [{"role": role_mapping.get(msg["role"], msg["role"]), "content": msg["content"]} for msg in processed_input]

        response = self.anthropic.messages.create(
            messages=messages,
            **self.model_params
        )

        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        return response.content[0].text, prompt_tokens, completion_tokens


class Claude35Sonnet(ClaudeBaseModel):
    def prompt(self, processed_input: list[dict], temperature=0.7, **kwargs):
        self.model_params.update(kwargs)
        self.model_params["temperature"] = temperature
        self.model_params["model"] = "claude-3-5-sonnet-20240620"
        return super().prompt(processed_input)


class Claude3Opus(ClaudeBaseModel):
    def prompt(self, processed_input: list[dict], temperature=0.7, **kwargs):
        self.model_params.update(kwargs)
        self.model_params["temperature"] = temperature
        self.model_params["model"] = "claude-3-opus-20240229"
        return super().prompt(processed_input)


class Claude3Haiku(ClaudeBaseModel):
    def prompt(self, processed_input: list[dict], temperature=0.7, **kwargs):
        self.model_params.update(kwargs)
        self.model_params["temperature"] = temperature
        self.model_params["model"] = "claude-3-haiku-20240307"
        return super().prompt(processed_input)