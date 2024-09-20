import os

from .MapCoder import MapCoder
from . import utils

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_custom.yaml')

NUM_STAGES = 3
class Custom(MapCoder):
    def __init__(self, k: int = 3, t: int = 5, temps=None, top_ps = None, *args, **kwargs):
        super().__init__(k, t, *args, **kwargs)
        self.prompts = utils.load_prompts(prompts_file)
        self.temps = [] if temps is None else temps
        self.top_ps = [] if top_ps is None else top_ps

    def run_single_pass(self, item: dict):
        print("", flush=True)

        # Step 1a: Generate KB exemplars
        self.update_temp_topp_param(0)
        input_kb_exemplars = self.create_kb_exemplars(item)
        utils.log("Input for knowledge base and exemplars: ", input_kb_exemplars[0]['content'])
        response, pr_tok, com_tok = self.gpt_chat(input_kb_exemplars, item)

        # Step 1b: Post process KB response
        self.update_temp_topp_param(1)
        response = self.post_process_response(response)
        utils.log("Response from knowledge base and exemplars: ", response)
        response = utils.parse_xml(response)
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"

        # Step 2: Generate plannings and confidences based on examples
        self.update_temp_topp_param(1)
        plannings, pr_tok, com_tok = self.generate_plannings(item, response, algorithm_prompt, sample_io_prompt, pr_tok,
                                                             com_tok)
        plannings.sort(key=lambda x: x[1], reverse=True)

        # Step 3: For each planning generate code. Iteratively improve code until it passes samples cases.
        self.update_temp_topp_param(2)
        code, pr_tok, com_tok = self.generate_final_code(item, plannings, algorithm_prompt, sample_io_prompt, pr_tok,
                                                         com_tok)

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok

    def update_temp_topp_param(self, stage):
        params = {}
        if len(self.temps) == NUM_STAGES:
            params["temperature"] = self.temps[stage]
        if len(self.top_ps) == NUM_STAGES:
            params["top_p"] = self.top_ps[stage]
        self.model.model_params.update(params)