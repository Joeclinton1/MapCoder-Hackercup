import math
import os

from .Matus import Matus
from . import utils
from ..results import write_debug
import re
import xml.etree.ElementTree as ET
import concurrent.futures
import threading
import json
import time


# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_joe.yaml')
algorithms_file = os.path.join(cwd, 'prompt_templates/algorithm_list.yaml')
lang_specific_file = os.path.join(cwd, 'prompt_templates/lang_specific_tips.yaml')

# constants that affect how much computation it will use
NUM_PARALLEL = 10
NUM_SETS = 1
NUM_TRICKS_PER_SET = 2
MAX_IMPROVEMENT_TRIES = 1
NUM_SHOTS = 8

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

        sol = dict(score=0.0, code="", plan="", test_report="")
        stop_event = threading.Event()  # Create an event to signal when to stop other parallel executions
        sol_lock = threading.Lock()  # Create a lock for updating the sol dictionary

        def single_shot(shot_index):
            nonlocal sol

            if stop_event.is_set():  # Check if the stop event has been triggered
                return None

            print(f"SHOT {shot_index}\n-----------------------------------")

            # Step 1: Generate tricks
            print(f"Generating {NUM_SETS} sets of {NUM_TRICKS_PER_SET} tricks for how to solve the problem \n")
            attempts = 0
            complexities, tricks = [], []
            while attempts < 3:
                complexities, tricks = self.generate_tricks(item, problem)
                if tricks:  # If tricks is not empty, break out of the loop
                    break
                attempts += 1
                print(f"Attempt {attempts} failed to generate tricks. Retrying...")

            if not tricks:
                raise Exception("Failed to generate tricks after 3 attempts.")

            # Step 2: Generate high-level plans
            print(f"Generating {NUM_SETS * NUM_TRICKS_PER_SET} plans for how to solve the problem \n")
            plans = self.generate_plans(item, problem, tricks)

            # Step 3: Generate codes for each plan and track best scores
            for i, (trick, plan) in enumerate(plans):
                if stop_event.is_set():  # Check if another thread has already solved the problem
                    return None

                print(f' --- Attempt {i} --- ')
                print(f'Trick: {trick}')
                score, code, test_report = self.generate_code(item, trick, plan, problem)

                with sol_lock:  # Lock the code block that modifies the shared `sol`
                    if score >= sol["score"]:
                        sol = dict(score=score, code=code, plan=plan, test_report=test_report)

                    if score == 1.0:
                        stop_event.set()  # Signal other threads to stop
                        return sol["code"], self.pr_tok, self.com_tok

                if score == 0.999:
                    break

            # Step 4: Try improving the best code seen so far
            print(f"--Best score so far: {sol['score']}--\n ## Improving best code: \n")
            for j in range(MAX_IMPROVEMENT_TRIES):
                if stop_event.is_set():  # Check if another thread has already solved the problem
                    return None

                score, code, test_report = self.improve_code(item, problem, sol['code'], sol["plan"],
                                                             sol["test_report"])

                with sol_lock:  # Lock the code block that modifies the shared `sol`
                    if score >= sol["score"]:
                        sol = dict(score=score, code=code, plan=sol["plan"], test_report=test_report)

                    if score == 1.0:
                        stop_event.set()  # Signal other threads to stop
                        return sol["code"], self.pr_tok, self.com_tok

            return None  # No successful result

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_SHOTS) as executor:
            futures = []
            for i in range(NUM_SHOTS):
                future = executor.submit(single_shot, i)
                futures.append(future)
                time.sleep(0.01)  # Sleep for 10 milliseconds between submissions to prevent timeout

            for future in concurrent.futures.as_completed(futures):
                result = future.result(timeout=600)
                if result is not None:
                    return result

        return sol["code"], self.pr_tok, self.com_tok

    def generate_tricks(self, item, problem):
        def parse_tricks(response):
            try:
                raw_xml = utils.replace_tag(response, 'trick')
                tree = utils.parse_xml_element(raw_xml)
                complexity = tree.find("complexity").text
                tricks = tree.find('tricks').findall('trick')
                return complexity, tricks
            except Exception as e:
                print(f"Error parsing XML: {e}")
                return "", []

        def gen_tricks(_):
            return parse_tricks(self.chat(trick_prompt, item, 'tricks', temperature=0.8))

        trick_prompt = self.prompts['trick'].format(
            problem_prompt=problem,
            algorithms_list=self.algorithms,
            num_tricks=NUM_TRICKS_PER_SET
        )
        write_debug(trick_prompt, 'trick_prompt')
        tricks = []
        complexities = []
        outputs = self.run_func_parallel_and_collect(gen_tricks, num_parallel=NUM_SETS)

        # Filter out outputs with empty trick lists
        outputs = [(complexity, tricks_xml) for complexity, tricks_xml in outputs if tricks_xml]

        print(" Complexity Targets:")
        for i, (complexity, tricks_xml) in enumerate(outputs):
            complexities.append(complexity)
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
        score, code, test_report = utils.holistic_get_best_result(results)

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {score}\n')

        return score, code, test_report

    def improve_code(self, item, problem, best_code, plan, test_result):
        print(" ## Modifying code")

        # Step 1. Create code improvement prompt.
        improvement_prompt = self.prompts['improve_plan_and_code_error'] \
            if "Error Type" in test_result \
            else self.prompts['improve_plan_and_code']
        improvement_prompt = improvement_prompt.format(
            problem_prompt=problem,
            planning=plan,
            language=self.language,
            test_log=test_result,
            code=best_code,
            lang_specific_tips=self.lang_specific_tips,
        )

        # Step 2: Generate NUM_PARALLEL//2+1 code modifications to fix test case
        def modify_code_and_evaluate(_):
            # Step 2a: Call the chat function to modify file
            response = self.chat(improvement_prompt, item, tag='improvement', temperature=0.9, top_p=1.0)
            code = utils.parse_code(response)

            # Step 3c: Reevaluate code_lines
            score, test_result_new = self.data.evaluate_sample_io(item, code, self.language)

            return score, code, test_result_new

        results = self.run_func_parallel_and_collect(modify_code_and_evaluate, num_parallel=NUM_PARALLEL)
        best_score, best_code, test_result = utils.holistic_get_best_result(results)

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {best_score}\n')

        return best_score, best_code, test_result

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

        # Step 1: generate code
        score, code, test_report = self.generate_code(item, "", plan, problem)
        sol = dict(score=score, code=code, plan=plan, test_report=test_report)

        # Step 2: Try improving the best code seen so far
        print(f"--Best score so far: {score}--\n ## Improving best code: \n")
        for j in range(MAX_IMPROVEMENT_TRIES):

            score, code, test_report = self.improve_code(item, problem, sol['code'], sol["plan"],
                                                         sol["test_report"])

            if score >= sol["score"]:
                sol = dict(score=score, code=code, plan=sol["plan"], test_report=test_report)

            if score == 1.0:
                return sol["code"], self.pr_tok, self.com_tok

        return code, self.pr_tok, self.com_tok

    def run_single_pass_code_improvement_only(self, item: dict, improvement_dict: dict, curr_pass:int):
        print(f"Testing wrong plan improvement #{curr_pass}")

        wrong_code = improvement_dict[f"wrong_code{curr_pass+1}"]
        wrong_plan = improvement_dict[f"wrong_plan{curr_pass+1}"]

        # add trick for fair comparison with MapCoder
        # trick = improvement_dict["trick"]
        # wrong_plan = f"## Trick\n{trick}\n{wrong_plan}"
        score, test_result = self.data.evaluate_sample_io(item, wrong_code, self.language)
        print(f"Starting Score: {score}")
        problem = self.data.get_prompt(item)
        _, code, _ = self.improve_code(item, problem, wrong_code, wrong_plan, test_result)

        return code, self.pr_tok, self.com_tok

    @staticmethod
    def run_func_parallel_and_collect(func, num_parallel=NUM_PARALLEL):
        return utils.run_func_parallel_and_collect(func, num_parallel)
