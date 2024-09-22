import os

from .Custom_DirectPlanning import DirectPlanning
from . import utils
from ..results import write_debug

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_custom.yaml')

NUM_STAGES = 2
class KShotCode(DirectPlanning):
    def __init__(self, k: int = 3, t: int = 5, temps=None, top_ps = None, *args, **kwargs):
        super().__init__(k, t, *args, **kwargs)
        self.prompts = utils.load_prompts(prompts_file)
        self.temps = [] if temps is None else temps
        self.top_ps = [] if top_ps is None else top_ps

    def run_single_pass(self, item: dict):
        print("", flush=True)

        # Step 1: Generate k plans directly
        self.update_temp_topp_param(1)
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"
        response = self.generate_plannings_directly(item, sample_io_prompt)
        for plan in response["planning"]:
            write_debug(dict(pseudocode=plan['pseudocode']), 'planning')
        algorithm_prompt = f""
        plannings = [x["pseudocode"] for x in response["planning"]]

        # Step 2: For each plan generate code. Iteratively improve code until it passes samples cases.
        self.update_temp_topp_param(2)
        best_score, best_code = 0.0, ""
        for planning in plannings:
            for i in range(self.t):
                score, code = self.generate_code_kshot(item, planning, sample_io_prompt,i)
                if score>=best_score:
                    best_score = score
                    best_code = code
                if score == 1:
                    break
        print("________________________\n\n", flush=True)
        return best_code, self.pr_tok, self.com_tok

    def generate_code_kshot(self, item,  planning, sample_io_prompt,i):
        input_for_code_gen = self.prompts['code_generation_input']['content'].format(
            language=self.language,
            algorithm_prompt='',
            problem_prompt=self.data.get_prompt(item),
            planning=planning,
            sample_io_prompt=sample_io_prompt,
            std_input_prompt=self.prompts['std_input_prompt']['content']
        )

        utils.log("Input for code generation:", input_for_code_gen)
        code = self.chat(input_for_code_gen, item)
        code = utils.parse_code(code)
        utils.log("Response from code generation: ", code)
        score, test_log = self.data.evaluate_sample_io(item, code, self.language)
        write_debug(dict(i=i, score=score, feedback=test_log, code=code), f'code_{i}')
        return score, code
