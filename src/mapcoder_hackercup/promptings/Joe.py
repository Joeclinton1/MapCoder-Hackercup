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
NUM_SETS = 3
NUM_TRICKS_PER_SET = 3
MAX_IMPROVEMENT_TRIES = 0

class Joe(Matus):
    def __init__(self, *args, **kwargs):
        self.prompts = utils.load_prompts(prompts_file)
        self.algorithms = utils.load_prompts(algorithms_file)['algorithm_list']
        self.pr_tok, self.com_tok = 0, 0
        super(Matus, self).__init__(*args, **kwargs)
        self.sample_io_prompt = None
        self.sample_explanation = None
        self.lang_specific_tips = utils.load_prompts(lang_specific_file)

    def run_single_pass(self, item):
        self.sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"
        problem = self.data.get_prompt(item)
        self.sample_explanation = problem.split("# sample explanation\n", 1)[-1]

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
                    break
                if score == prev_score:
                    print(f'Score is not improving, stopping...')
                    break

                code, score, test_result = self.fix_code_lines_failing_test_case(item, problem, max_code, test_report)

        return max_code, self.pr_tok, self.com_tok

    def generate_tricks(self, item, problem):
        def parse_tricks(response):
            raw_xml = utils.replace_tag(response, 'trick')
            tree = utils.parse_xml_element(raw_xml)
            return tree.find("complexity"), tree.find('tricks').findall('trick')

        trick_prompt = self.prompts['trick'].format(
            problem_prompt=problem,
            algorithms_list=self.algorithms,
            num_tricks=NUM_TRICKS_PER_SET
        )
        write_debug(trick_prompt, 'trick_prompt')
        tricks = []
        complexities = []
        print(" Complexity Targets:")
        for i in range(NUM_SETS):
            complexity, tricks_xml = parse_tricks(self.chat(trick_prompt, item, 'tricks', temperature=0.8))
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

        lang_specific = f"# Language specific tips:\n{self.lang_specific_tips[self.language]}" \
            if self.language in self.lang_specific_tips else ""
        code_prompt = self.prompts['coding'].format(
            problem_prompt=problem_prompt,
            trick=trick,
            planning = plan,
            sample_io_prompt=self.sample_io_prompt,
            language=self.language,
            lang_specific_tips=lang_specific
        )

        def gen_code(i):
            code_output = self.chat(code_prompt, item, tag='code', temperature=0.5)
            code = utils.parse_code(code_output)
            score, test_result = self.data.evaluate_sample_io(item, code, self.language)
            return score, code, test_result

        write_debug(code_prompt, 'code_prompt')
        results = self.run_func_parallel_and_collect(gen_code, num_parallel=NUM_PARALLEL)
        results2 = [x for x in results if not (isinstance(x[0], int) and x[0] == 0)]
        if len(results2) == 0:
            results2 = results
        score, code, test_report = max(results2, key=lambda x: x[0])

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {score}\n')

        return score, code, test_report

    def fix_code_lines_failing_test_case(self, item, problem, best_code, test_result):
        print(" ## Modifying code")

        # Step 1. Get the expected and actual output of the test case that failed
        expected_actual = self.get_first_failed_case(test_result)

        # Step 2: Add line numbers to the code, so the llm can reference them during debugging
        code_lines = best_code.split('\n')
        numbered_code = '\n'.join([f"{i}: {line}" for i, line in enumerate(code_lines)])

        # Step 3: Generate NUM_PARALLEL//2+1 code modifications to fix test case
        def modify_code_and_evaluate(_):
            # Step 3a: Call the chat function to get modified lines
            mod_reasoning, mod_lines = self.modify_code_lines(problem, expected_actual, item, numbered_code)
            self.log_modification(mod_reasoning, mod_lines, code_lines)

            # Step 3b: Replace original code file lines with modified code lines and join
            for num, new in mod_lines:
                old = code_lines[num]
                indent = old[:len(old) - len(old.lstrip())]
                code_lines[num] = indent + new.lstrip()
            code = "\n".join(code_lines)

            write_debug(dict(code=code), type_='code')

            # Step 3c: Reevaluate code_lines
            score, test_result_new = self.data.evaluate_sample_io(item, code, self.language)

            return score, code, test_result_new

        results = self.run_func_parallel_and_collect(modify_code_and_evaluate, num_parallel=NUM_PARALLEL//2+1)
        best_score, best_code, test_result = max(results, key=lambda x: x[0])

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {best_score}\n')

        return best_code, best_score, test_result

    def modify_code_lines(self, problem, expected_actual, item, numbered_code):
        modify_prompt = self.prompts['modify'].format(
            problem_prompt=problem,
            sample_io_prompt=self.sample_io_prompt,
            language=self.language,
            expected=expected_actual[0],
            actual=expected_actual[1],
            numbered_code=numbered_code
        )
        mod_raw_xml = self.chat(modify_prompt, item, tag='modify_lines', temperature=0.2, top_p=1.0)
        mod_raw_xml = utils.replace_tag(mod_raw_xml, 'num')
        mod_raw_xml = utils.replace_tag(mod_raw_xml, 'code')
        mod_xml = utils.parse_xml_element(mod_raw_xml)
        mod_reasoning = mod_xml.find("explanation").text
        mod_lines = mod_xml.findall('change')
        mod_lines = [
            [int(l.find('num').text.strip()), l.find('code').text]
            for l in mod_lines
        ]
        return mod_reasoning, mod_lines

    def log_modification(self, mod_reasoning, mod_lines, code_lines):
        logging_dict = {'explanation': '\n'+mod_reasoning.strip()}
        print(f"Modification Explanation: {mod_reasoning}")
        for num, new in mod_lines:
            logging_dict[f'line_{num}'] = f'\nOld: {code_lines[num]}\nNew: {new}'
        write_debug(logging_dict, 'modify')

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
        code, _, _ = self.fix_code_lines_failing_test_case(item, problem, code, test_result)
        return code, self.pr_tok, self.com_tok