import os

from .MapCoder import MapCoder
from . import utils
from ..results import write_debug

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_custom.yaml')

NUM_STAGES = 3
class Custom(MapCoder):
    def __init__(self, k: int = 3, t: int = 5, *args, **kwargs):
        super().__init__(k, t, *args, **kwargs)
        self.prompts = utils.load_prompts(prompts_file)

    def run_single_pass(self, item: dict):
        print("", flush=True)
        # # Step 0: Improve problem prompt
        # self.improve_problem_prompt(item)

        # Step 1: Generate KB exemplars and algorithm
        self.update_temp_topp_param(0)
        response = self.generate_kb_exemplars_and_algorithm(item)

        # Step 2: Generate plannings based on examples and sort by confidence
        self.update_temp_topp_param(1)
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        plannings = self.generate_plannings(item, response, algorithm_prompt, sample_io_prompt)
        plannings = [x[0] for x in sorted(plannings, key=lambda x: x[1], reverse=True)]

        # Step 3: For each planning generate code. Iteratively improve code until it passes samples cases.
        self.update_temp_topp_param(2)
        code = self.generate_final_code(item, plannings, algorithm_prompt, sample_io_prompt)

        print("________________________\n\n", flush=True)
        return code, self.pr_tok, self.com_tok

    def update_temp_topp_param(self, stage):
        params = {}
        if len(self.temps) == NUM_STAGES:
            params["temperature"] = self.temps[stage]
        if len(self.top_ps) == NUM_STAGES:
            params["top_p"] = self.top_ps[stage]
        self.model.model_params.update(params)

    def improve_problem_prompt(self, item):
        input_for_problem_prompt_rewrite = self.prompts['problem_prompt_rewrite']['content'].format(
            problem_prompt=self.data.get_prompt(item),
        )

        improved_problem = self.chat(input_for_problem_prompt_rewrite, item)
        write_debug(dict(improved_problem=improved_problem), "rewrite_prompt")
        item["description"] = improved_problem


    def run_single_pass_no_planning(self, item: dict, plan: str):
        # self.improve_problem_prompt(item)
        self.update_temp_topp_param(2)
        return super().run_single_pass_no_planning(item, plan)
