import os

from .Custom import Custom
from . import utils
from ..results import write_debug

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_custom.yaml')
prompts_file_pai = os.path.join(cwd, 'prompt_templates/prompts_directplanning.yaml')

NUM_STAGES = 2
class DirectPlanning(Custom):
    def __init__(self, k: int = 3, t: int = 5, temps=None, top_ps = None, *args, **kwargs):
        super().__init__(k, t, *args, **kwargs)
        self.prompts = utils.load_prompts(prompts_file)
        self.prompts2 = utils.load_prompts(prompts_file_pai)
        self.temps = [] if temps is None else temps
        self.top_ps = [] if top_ps is None else top_ps

    def run_single_pass(self, item: dict):
        print("", flush=True)

        self.update_temp_topp_param(1)

        # Step 1: Generate k plans directly
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"
        response = self.generate_plannings_directly(item, sample_io_prompt)
        for plan in response["planning"]:
            write_debug(dict(pseudocode=plan['pseudocode']), 'planning')
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        write_debug(dict(algorithm=algorithm_prompt), 'algorithm')
        plannings = [x["pseudocode"] for x in response["planning"]]

        # Step 2: For each plan generate code. Iteratively improve code until it passes samples cases.
        self.update_temp_topp_param(2)
        code = self.generate_final_code(item, plannings, algorithm_prompt, sample_io_prompt)

        print("________________________\n\n", flush=True)
        return code, self.pr_tok, self.com_tok

    def generate_plannings_directly(self, item, sample_io_prompt):
        plan_gen_input = self.prompts2['generate_plans']['content'].format(
            problem_prompt=self.data.get_prompt(item),
            sample_io_prompt=sample_io_prompt,
            mapping_k=self.k,
        )
        utils.log("Input for plan generation: ", plan_gen_input)
        response = self.chat(plan_gen_input, item)
        response = self.post_process_planning_response(response)

        utils.log("Response from our problem planning: ", response)
        response = utils.parse_xml(response)
        return response

    def post_process_planning_response(self, response):
        response = utils.trim_text(response, "# Identify the algorithm")
        response = utils.trim_text(response, "# Write a useful tutorial")
        response = utils.trim_text(response, "# Pseudocode planning to solve this problem")
        response = utils.replace_tag(response, 'pseudocode')
        return response

