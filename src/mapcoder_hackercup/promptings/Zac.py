
# plans I have:
# Deduction - at the moment we generate all plans independently
# I wonder about if we could be simultaneusly generating plans, working on them, and whilst promising solutions are fixed, 
# we re-generate plans with the prompt including failed approaches so far.
# We could keep track of how many workers we have in parallel, and if any are free we can assign them to regenerate plans w/ deduction

# Or, we could also do more "exploration" when a solution is nearly correct.
# So, we would be developing this tree where liekly branches are exploited and then exploration happens if stuck
# Simultaneusly, we continue exploring alternative branches 

# Another plan I have is to tell the LLM what level hardness the problem is, so for Q1/2 we expect to find simpler algorithms with much nicer time complexity (so it shouldn't overthink it)

import os

from .Matus import Matus
from . import utils
import concurrent.futures
from ..results import write_debug
import re
import xml.etree.ElementTree as ET
from typing import List, Dict

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_zac.yaml')
lang_specific_file = os.path.join(cwd, 'prompt_templates/lang_specific_tips.yaml')

NUM_PARALLEL = 7
NUM_SETS = 2
NUM_PLANS_PER_SET = 3
MAX_IMPROVEMENT_TRIES = 1
THRESH_FOR_IMPROVEMENT = 0.5

class Zac(Matus):
    def __init__(self, *args, **kwargs):
        self.prompts = utils.load_prompts(prompts_file)
        # self.algorithms = utils.load_prompts(algorithms_file)['algorithm_list']
        self.pr_tok, self.com_tok = 0, 0
        super(Matus, self).__init__(*args, **kwargs)
        self.sample_io_prompt = None

        tips = utils.load_prompts(lang_specific_file)
        self.lang_specific_tips = f"# Language specific tips:\n{tips[self.language]}" if self.language in tips else ""

    def run_single_pass(self, item):
        self.sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"
        problem = self.data.get_prompt(item)

        sol = dict(score=0.0, code="", plan="", test_report="")
        # num_shots = round(14//NUM_PARALLEL)
        num_shots = 2

        for i in range(num_shots):
            print(f"SHOT {i}\n-----------------------------------")
            # Step 1: Generate k plans
            print(f"Generating {NUM_SETS} sets of {NUM_PLANS_PER_SET} plans for how to solve the problem \n")
            all_plans = self.generate_plans(item, problem)

            # Skipping this 
            # Step 2: For each plan generate a high level plan
            # print(f"Generating {NUM_SETS*NUM_planS_PER_SET} plans for how to solve the problem \n")
            # plans = self.generate_plans(item, problem, plans)

            # Step 3: Generate NUM_PARALLEL codes for each plan in parallel and keep track of best scoring codes
            for plan_set in all_plans: 
                
                for i, (plan_dict) in enumerate(plan_set):

                    plan = plan_dict['explanation']
                    complexity = plan_dict['complexity']

                    print(f' --- Attempt {i} --- ')
                    print(f'plan: {plan}')

                    score, code, test_report = self.generate_code(item, plan, complexity, problem)

                    if score >= sol["score"]:
                        sol = dict(score=score, code=code, plan=plan, test_report=test_report)
                    if score == 1.0:
                        return sol["code"], self.pr_tok, self.com_tok
                    if score == 0.999:
                        break

            print(f"--Best score so far: {sol['score']}--\n ## Improving best code: \n")

            # Step 4: For the best scoring plan seen so far. Improve its results.
            for j in range(MAX_IMPROVEMENT_TRIES):
                score, code, test_report = self.improve_code(
                    item, problem, sol['code'], sol["plan"], sol["test_report"]
                )

                if score >= sol["score"]:
                    sol = dict(score=score, code=code, plan=sol["plan"], test_report=test_report)

                if score == 1.0:
                    return sol["code"], self.pr_tok, self.com_tok

            # Step 5: do the entire NUM_SETS x NUM_planS_PER_SET number of plan attempts again

        return sol["code"], self.pr_tok, self.com_tok

    def generate_plans(self, item, problem) -> List[List[Dict]]:
        def parse_plans(response):
            
            # List of dicts, with 'explanation' and 'complexity' keys in each. 
            plans_list = [] 

            # raw_xml = utils.replace_tag(response, 'plan') # not sure what this does, it broke my parsing 
            # tree = utils.parse_xml_element(raw_xml)

            tree = utils.parse_xml_element(response)
        
            # Find all <plan> elements under <plans>
            plans = tree.find('plans').findall('plan') # list of plans 

            # Loop through each <plan> and extract the <explanation> and <complexity> fields
            for plan in plans:
                title = plan.find('title').text.strip() 
                explanation = plan.find('explanation').text.strip()
                complexity = plan.find('complexity').text.strip()
                plans_list.append({
                    'explanation': title + ': ' + explanation,
                    'complexity': complexity
                })

            return plans_list 

        def gen_plans(_):
            return parse_plans(self.chat(plan_prompt, item, 'plans', temperature=0.8))

        plan_prompt = self.prompts['hard_plans'].format(
            problem_prompt=problem,
            num_plans=NUM_PLANS_PER_SET
        )
        write_debug(plan_prompt, 'plan_prompt')

        outputs = self.run_func_parallel_and_collect(gen_plans, num_parallel=NUM_SETS)
        
        return outputs 


    def generate_code(self, item, plan, complexity, problem_prompt):
        print(f" Generating Code:")

        code_prompt = self.prompts['coding'].format(
            problem=problem_prompt,
            plan=plan,
            complexity=complexity,
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
        score, code, test_report = self.holistic_get_best_result(results)

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {score}\n')

        return score, code, test_report


    def improve_code(self, item, problem, best_code, plan, test_result):
        print(" ## Modifying code")

        improvement_prompt = self.prompts['improve_plan_and_code']

        improvement_prompt = improvement_prompt.format(
            problem_prompt=problem,
            plan=plan, # i.e., plan 
            language=self.language,
            test_log=test_result,
            code=best_code,
            lang_specific_tips=self.lang_specific_tips,
        )

        write_debug(improvement_prompt, 'code_prompt')

        # Step 2: Generate NUM_PARALLEL//2+1 code modifications to fix test case
        def modify_code_and_evaluate(_):
            # Step 2a: Call the chat function to modify file
            response = self.chat(improvement_prompt, item, tag='improvement', temperature=0.9, top_p=1.0)
            code = utils.parse_code(response)

            # Step 3c: Reevaluate code_lines
            score, test_result_new = self.data.evaluate_sample_io(item, code, self.language)

            return score, code, test_result_new

        results = self.run_func_parallel_and_collect(modify_code_and_evaluate, num_parallel=NUM_PARALLEL)
        best_score, best_code, test_result = self.holistic_get_best_result(results)

        print(f' Scores: {",".join([str(r[0]) for r in results])}')
        print(f' Best Score: {best_score}\n')

        return best_score, best_code, test_result

    @staticmethod
    def holistic_get_best_result(results):
        # Instead of max score being returned use the average of the top two scores.
        results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_code, test_result = results[0]
        if best_score == 1.0 or len(results) == 1:
            return best_score, best_code, test_result

        # if the best score == 0, then it might be 0 meaning error or 0.0 meaning just wrong answer
        # a wrong answer solution is better than error so we will remove the 0 solutions
        if best_score == 0:
            results2 = [x for x in results if not (isinstance(x[0], int) and x[0] == 0)]
            if len(results2) == 0:
                results2 = results
            best_score, best_code, test_result = results2[0]

        # weighted holistic scoring so that the second-best score is taken into account
        # intuition is that if the second-best score is low but top is high then the plan is not actually good
        average_top_two_score = results[0][0]*0.6+results[1][0]*0.4
        return average_top_two_score, best_code, test_result

    @staticmethod
    def run_func_parallel_and_collect( func, num_parallel=NUM_PARALLEL):
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






