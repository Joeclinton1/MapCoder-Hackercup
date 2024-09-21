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
        self.n_improvements = kwargs.get('n_improvements', 12)

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
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"

        max_score, max_code = 0.0, ""
        plan_hist = []

        for i in range(self.n_plans):
            print(f' --- Attempt {i} --- ')
            print(f' Generating plan ')

            plan_hist_str = ""
            if plan_hist:
                plan_hist_str = self.prompts['breakdown']['plan_hist_desc'] + \
                '\n'.join((self.prompts['breakdown']['separator'].format(idx=idx + 1, plan=plan)
                    for idx, plan in enumerate(plan_hist)))

            planning_prompt = self.prompts['breakdown']['content'] \
                .format(problem_prompt=problem_prompt, history=plan_hist_str)

            plan = self.chat(planning_prompt, item, 'breakdown')
            plan_hist.append(
                self.chat(
                    self.prompts['breakdown_history']['content']
                    .format(plan=plan),
                    item=item,
                    tag='summary'
                )
            )

            code_prompt = self.prompts['coding']['content']\
                .format(problem_prompt=problem_prompt,
                        plan=plan,
                        sample_io_prompt=sample_io_prompt,
                        std_input_prompt=self.prompts['std_input_prompt']['content'],
                        language=self.language)

            code_output = self.chat(code_prompt, item, tag='code')
            for i in range(self.n_improvements):
                code = utils.parse_code(code_output)
                write_debug(code, 'code')

                score, test_result = self.data.evaluate_sample_io(item, code, self.language)
                print(f' Attempt {i + 1}, score {score}')

                if score == 1.0:
                    return code, self.pr_tok, self.com_tok

                if score > max_score:
                    max_score, max_code = score, code

                critique_prompt = self.prompts['critique']['content'] \
                    .format(problem_prompt=problem_prompt,
                            code=code,
                            test_log=test_result,
                            std_input_prompt=self.prompts['std_input_prompt']['content'],
                            language=self.language)
                critique = self.chat(critique_prompt, item, 'code')

                # improvement_prompt = self.prompts['improvement']['content'] \
                #     .format(critique=critique,
                #             code=code,
                #             test_log=test_result,
                #             std_input_prompt=self.prompts['std_input_prompt']['content'],
                #             language=self.language)

                code_output = utils.parse_code(critique)  # self.chat(critique, item, tag='code'))

        return max_code, self.pr_tok, self.com_tok
