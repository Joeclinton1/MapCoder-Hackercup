import os

from .MapCoder import MapCoder
from . import utils

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_custom.yaml')


class Custom(MapCoder):
    def __init__(self, k: int = 3, t: int = 5, *args, **kwargs):
        super().__init__(k, t, *args, **kwargs)
        self.prompts = utils.load_prompts(prompts_file)

    def run_single_pass(self, item: dict):
        print("", flush=True)
        input_kb_exemplars = self.create_kb_exemplars(item)
        response, pr_tok, com_tok = self.gpt_chat(input_kb_exemplars)
        item['api_calls'] = item.get('api_calls', 0) + 1

        response = self.post_process_response(response)
        print("\n\n________________________")
        print("Response from knowledge base and exemplars: ")
        print(response, flush=True)

        response = utils.parse_xml(response)
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"

        # Generate plannings based on examples
        plannings, pr_tok, com_tok = self.generate_plannings(item, response, algorithm_prompt, sample_io_prompt, pr_tok,
                                                             com_tok)

        # Sort plannings by confidence and generate code
        plannings.sort(key=lambda x: x[1], reverse=True)
        code, pr_tok, com_tok = self.generate_final_code(item, plannings, algorithm_prompt, sample_io_prompt, pr_tok,
                                                         com_tok)

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok