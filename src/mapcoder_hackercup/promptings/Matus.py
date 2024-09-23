import os

from tqdm import tqdm
from . import utils
from .Base import BaseStrategy
from ..results import write_debug

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_matus.yaml')


class Matus(BaseStrategy):
    def __init__(self, *args, **kwargs):
        self.prompts = utils.load_prompts(prompts_file)
        self.n_plans = kwargs.get('n_plans', 5)
        self.n_improvements = kwargs.get('n_improvements', 8)

        self.pr_tok, self.com_tok = 0, 0

        super(Matus, self).__init__(*args, **kwargs)

    def chat(self, input: str, item: dict, tag='', **kwargs) -> (str, int, int):
        item['api_calls'] = item.get('api_calls', 0) + 1
        response, pr_tok, com_tok = self.model.prompt(
            processed_input=[{"role": "user", "content": input}],
            **kwargs
        )

        self.pr_tok += pr_tok
        self.com_tok += com_tok

        write_debug(input, tag + '_' + 'prompt')
        write_debug(response, tag)
        return response

    def run_single_pass(self, item):
        problem_prompt = self.data.get_prompt(item)
        planning_prompt = self.prompts['breakdown']['content']\
            .format(problem_prompt=problem_prompt)
        sample_io_prompt=f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"

        max_score, max_code = 0.0, ""

        for i in range(self.n_plans):
            print(f' --- Attempt {i} --- ')
            print(f' Generating plan ')
            plan = self.chat(planning_prompt, item, 'breakdown')
            max_score, max_code = self.generate_code(
                item, plan, max_score, max_code, problem_prompt, sample_io_prompt
            )
            if max_score == 1.0:
                break

        return max_code, self.pr_tok, self.com_tok

    def generate_code(self, item, plan, max_score, max_code, problem_prompt, sample_io_prompt):
        code_prompt = self.prompts['coding']['content'] \
            .format(problem_prompt=problem_prompt,
                    plan=plan,
                    sample_io_prompt=sample_io_prompt,
                    std_input_prompt=self.prompts['std_input_prompt']['content'],
                    language=self.language)

        code_output = self.chat(code_prompt, item, tag='code')
        for i in range(self.n_improvements):
            code = utils.parse_code(code_output)
            score, test_result = self.data.evaluate_sample_io(item, code, self.language)
            print(f' Attempt {i + 1}, score {score}')

            if score == 1.0:
                return 1.0, code

            if score > max_score:
                max_score, max_code = score, code

            critique_prompt = self.prompts['critique']['content'] \
                .format(problem_prompt=problem_prompt,
                        code=code,
                        test_log=test_result,
                        language=self.language)
            critique = self.chat(critique_prompt, item, 'critique')

            improvement_prompt = self.prompts['improvement']['content'] \
                .format(critique=critique,
                        code=code,
                        test_log=test_result,
                        std_input_prompt=self.prompts['std_input_prompt']['content'],
                        language=self.language)

            code_output = utils.parse_code(self.chat(improvement_prompt, item, tag='code'))
        return max_score, max_code

    def run_single_pass_no_planning(self, item: dict, plan: str):
        problem_prompt = self.data.get_prompt(item)
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"
        write_debug(plan, "plan")
        _, code = self.generate_code(item, plan, 0.0, "", problem_prompt, sample_io_prompt)
        return code, self.pr_tok, self.com_tok
