import os

from .Matus import Matus
from . import utils
import concurrent.futures
from ..results import write_debug
import re
import xml.etree.ElementTree as ET

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_joe.yaml')
algorithms_file = os.path.join(cwd, 'prompt_templates/algorithm_list.yaml')
lang_specific_file = os.path.join(cwd, 'prompt_templates/lang_specific_tips.yaml')

NUM_PARALLEL = 5
NUM_SETS = 4
NUM_TRICKS_PER_SET = 2
MAX_IMPROVEMENT_TRIES = 4
THRESH_FOR_IMPROVEMENT = 0.5

class Joe(Matus):
    def __init__(self, *args, **kwargs):
        self.prompts = utils.load_prompts(prompts_file)
        self.algorithms = utils.load_prompts(algorithms_file)['algorithm_list']
        self.pr_tok, self.com_tok = 0, 0
        super(Matus, self).__init__(*args, **kwargs)
        self.sample_io_prompt = None

        tips = utils.load_prompts(lang_specific_file)
        self.lang_specific_tips = f"# Language specific tips:\n{tips[self.language]}" if self.language in tips else ""

    def run_single_pass(self, item):
        self.sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"
        problem = self.data.get_prompt(item)

        # Step 1: Generate k tricks
        print(f"Generating {NUM_SETS} sets of {NUM_TRICKS_PER_SET} tricks for how to solve the problem \n")
        complexities, tricks = self.generate_tricks(item, problem)

        # Step 2: For each trick generate a high level plan
        print(f"Generating {NUM_SETS*NUM_TRICKS_PER_SET} plans for how to solve the problem \n")
        plans = self.generate_plans(item, problem, tricks)

        # Step 3: Generate NUM_PARALLEL codes for each plan in parallel and keep track of best scoring codes
        max_score, max_code = 0.0, ""
        for i, (trick, plan) in enumerate(plans):

            # Step 3a: Generate initial code
            print(f' --- Attempt {i} --- ')
            print(f'Trick: {trick}')
            score, code, test_report = self.generate_code(item, trick, plan, problem)

            prev_score = None
            # Step 3b: Improve code on test cases
            for j in range(MAX_IMPROVEMENT_TRIES):
                if score >= max_score:
                    max_score, max_code = score, code

                if score == 1.0:
                    return max_code, self.pr_tok, self.com_tok

                if score <= 0.5:
                    break
                if prev_score is not None and score <= prev_score:
                    print(f'Score is not improving, stopping...')
                    break

                prev_score = score
                score, code, test_result = self.improve_code(item, problem, max_code, test_report)

        return max_code, self.pr_tok, self.com_tok

    def generate_tricks(self, item, problem):
        def parse_tricks(response):
            raw_xml = utils.replace_tag(response, 'trick')
            tree = utils.parse_xml_element(raw_xml)
            return tree.find("complexity"), tree.find('tricks').findall('trick')

        def gen_tricks(_):
            return  parse_tricks(self.chat(trick_prompt, item, 'tricks', temperature=0.8))

        trick_prompt = self.prompts['trick'].format(
            problem_prompt=problem,
            algorithms_list=self.algorithms,
            num_tricks=NUM_TRICKS_PER_SET
        )
        write_debug(trick_prompt, 'trick_prompt')
        tricks = []
        complexities = []
        outputs = self.run_func_parallel_and_collect(gen_tricks, num_parallel=NUM_SETS)

        print(" Complexity Targets:")
        for i, (complexity, tricks_xml) in enumerate(outputs):
            complexities.append(complexity.text)
            print(f"Set {i}: {complexities[-1]}")
            tricks.extend([t.text for t in tricks_xml])
        return complexities, tricks

    def generate_plans(self, item, problem, tricks):
        def gen_plan(i):
            plan_prompt = self.prompts['planning'].format(
                problem_prompt=problem,
                sample_io_prompt=self.sample_io_prompt,
                trick=tricks[i]
            )
            return [tricks[i], self.chat(plan_prompt, item, 'plan', temperature=0.7)]

        plans = self.run_func_parallel_and_collect(gen_plan ,num_parallel=len(tricks))
        return plans

    def generate_code(self, item, trick, plan, problem_prompt):
        print(f" Generating Code:")


        code_prompt = self.prompts['coding'].format(
            problem_prompt=problem_prompt,
            trick=trick,
            planning = plan,
            sample_io_prompt=self.sample_io_prompt,
            language=self.language,
            lang_specific_tips=self.lang_specific_tips
        )

        def gen_code(i):
            code_output = self.chat(code_prompt, item, tag='code', temperature=0.5)
            code = utils.parse_code(code_output)
            score, test_result = self.data.evaluate_sample_io(item, code, self.language)
            return score, code, test_result

        write_debug(code_prompt, 'code_prompt')
        results = self.run_func_parallel_and_collect(gen_code)
        results2 = [x for x in results if not (isinstance(x[0], int) and x[0] == 0)]
        if len(results2) == 0:
            results2 = results
        score, code, test_report = max(results2, key=lambda x: x[0])

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {score}\n')

        return score, code, test_report

    def improve_code(self, item, problem, best_code, test_result):
        print(" ## Modifying code")

        # Step 1. if case failed with wrong answer,
        # get the expected and actual output of the test case that failed and generate prompt with it
        # else if it failed with error generate a prompt asking to fix this error
        if "Error Type" in test_result:
            improvement_prompt = self.prompts['code_improvement_error'].format(
                problem_prompt=problem,
                sample_io_prompt=self.sample_io_prompt,
                language=self.language,
                error=test_result,
                code=best_code,
                lang_specific_tips=self.lang_specific_tips,
            )
        else:
            expected, actual = self.get_first_failed_case(test_result)

            improvement_prompt = self.prompts['code_improvement'].format(
                problem_prompt=problem,
                sample_io_prompt=self.sample_io_prompt,
                language=self.language,
                expected=expected,
                actual=actual,
                code=best_code,
                lang_specific_tips=self.lang_specific_tips
            )

        # Step 2: Generate NUM_PARALLEL//2+1 code modifications to fix test case
        def modify_code_and_evaluate(_):
            # Step 2a: Call the chat function to modify file
            response = self.chat(improvement_prompt, item, tag='improvement', temperature=0.9, top_p=1.0)
            code = utils.parse_code(response)

            # Step 3c: Reevaluate code_lines
            score, test_result_new = self.data.evaluate_sample_io(item, code, self.language)

            return score, code, test_result_new

        results = self.run_func_parallel_and_collect(modify_code_and_evaluate, num_parallel=5) # // 2 + 1
        best_score, best_code, test_result = max(results, key=lambda x: x[0])

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {best_score}\n')

        return best_score, best_code, test_result

    @staticmethod
    def get_first_failed_case(test_result):
        result_lines = test_result.split('\n')
        i,j = result_lines.index('Expected Output:')+1, result_lines.index('Your output:')+1
        num_cases = j-i
        for idx in range(num_cases):
            if result_lines[i + idx] != result_lines[j + idx]:
                return result_lines[i + idx], result_lines[j + idx]

    def run_func_parallel_and_collect(self, func, num_parallel=NUM_PARALLEL):
        # Running the code generation in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
            futures = [executor.submit(func, i) for i in range(num_parallel)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return results

    def chat(self, input: str, item: dict, tag='', **kwargs) -> (str, int, int):
        item['api_calls'] = item.get('api_calls', 0) + 1
        response, pr_tok, com_tok = self.model.prompt(
            processed_input=[{"role": "user", "content": input}],
            **kwargs
        )
        self.pr_tok += pr_tok
        self.com_tok += com_tok
        write_debug(response, tag)
        return response

    def run_single_pass_no_planning(self, item: dict, plan: str):
        self.sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"
        problem = self.data.get_prompt(item)
        score, code = self.generate_code(item, "", plan, problem)
        return code, self.pr_tok, self.com_tok

    def run_single_pass_code_improvement_only(self, item: dict, code_dir: str):
        with open(code_dir, 'r') as f:
            code = f.read()
        score, test_result = self.data.evaluate_sample_io(item, code, self.language)
        print(f"Starting Score: {score}")
        problem = self.data.get_prompt(item)
        code, _, _ = self.improve_code(item, problem, code, test_result)
        return code, self.pr_tok, self.com_tok