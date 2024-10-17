from typing import List
import tiktoken
import os
import copy
import time
import json
import traceback

from mapcoder_hackercup.models.Base import BaseModel
from mapcoder_hackercup.datasets.Dataset import Dataset
from mapcoder_hackercup.results.Results import Results
from mapcoder_hackercup.utils.parse import parse_response
from mapcoder_hackercup.gen_comp_out import output_results
from . import utils



class BaseStrategy(object):
    def __init__(
        self,
        model: BaseModel,
        data: Dataset,
        language: str,
        pass_at_k: int,
        results: Results,
        verbose: bool = True,
        temps: list = None,
        top_ps: list = None,
        plan: str = None,
        improvement_dir: str = None
    ):
        self.model = model
        self.data = data
        self.pass_at_k = pass_at_k
        self.results = results
        self.language = language
        self.verbose = verbose
        self.temps = [] if temps is None else temps
        self.top_ps = [] if top_ps is None else top_ps
        self.plan = plan
        self.improvement_dir = improvement_dir

        # Test two improvements on two separate mistakes
        if self.improvement_dir is not None:
            self.pass_at_k = 2

    def gpt_chat(self, processed_input: List[dict], **kwargs) -> (str, int, int):
        return self.model.prompt(processed_input=processed_input, **kwargs)

    def run_single_pass(self, item: dict):
        pass

    def run_single_pass_no_planning(self, item: dict, plan: str):
        return self.run_single_pass(item)

    def run_single_pass_code_improvement_only(self, item: dict, improvement_dict: dict, curr_pass: int):
        return self.run_single_pass(item)

    def run(self):
        num_items = len(self.data)
        num_success = 0

        for i, item in enumerate(self.data):
            print("", flush=True, end="")

            if i < len(self.results):
                item = copy.deepcopy(self.results[i])
                cur_pass = len(item["source_codes"])
                is_solved = item["is_solved"]
                output = item["output"]
                cur_imp = item["source_codes"][-1]
            else:
                item = copy.deepcopy(item)
                item["source_codes"] = []
                item["responses"] = []
                item["prompt_tokens"] = []
                item["completion_tokens"] = []
                item["no_of_try"] = 0
                item["is_solved_sample"] = []
                item["sample_actual_output"] = []

                cur_pass = 0
                is_solved = False
                output = ""
                cur_imp = ""

            # generate a scorer which if set will be used instead of matching during evaluate
            self.generate_scorer(item)
            while cur_pass < self.pass_at_k and not is_solved:
                for _ in range(10):
                    try:
                        if self.plan is not None:
                            response, prompt_tokens, completion_tokens = self.run_single_pass_no_planning(item, self.plan)
                        elif self.improvement_dir is not None:
                            # Open and read the JSON file
                            with open(self.improvement_dir, 'r') as file:
                                improvement_data = json.load(file)

                            if item[self.data.id_key] in improvement_data:
                                improvement_data = improvement_data[item[self.data.id_key]]
                            else:
                                print(f"'{item[self.data.id_key]}' not found in data file. SKIPPING.")
                                break

                            result = self.run_single_pass_code_improvement_only(item, improvement_data, cur_pass)
                            response, prompt_tokens, completion_tokens = result
                        else:
                            response, prompt_tokens, completion_tokens = self.run_single_pass(item)
                        break
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        traceback.print_exc()  # This will print the full stack trace
                        time.sleep(5)
                        pass

                if response is None:
                    print('No valid code generated!')

                if hasattr(self, "parse_code"):
                    cur_imp = self.parse_code(response)
                else:
                    cur_imp = parse_response(response)
                    # cur_imp = parse_response(response, item.get("entry_point", None))

                item["source_codes"].append(cur_imp)
                item["responses"].append(response)
                item["prompt_tokens"].append(prompt_tokens)
                item["completion_tokens"].append(completion_tokens)
                item["no_of_try"] += 1

                if callable(getattr(self.data, "evaluate_sample_io", None)):
                    is_solved_sample, sample_actual_output = self.data.evaluate_sample_io(
                        item=item,
                        cur_imp=cur_imp,
                        language=self.language
                    )
                    item["is_solved_sample"].append(is_solved_sample)
                    item["sample_actual_output"].append(sample_actual_output)


                is_solved, output = self.data.evaluate(
                    item=item,
                    cur_imp=cur_imp,
                    language=self.language
                )

                cur_pass += 1

            if is_solved:
                num_success += 1

            item["is_solved"] = is_solved
            item["language"] = self.language
            item["task_id"] = item[self.data.id_key]
            # item["full_output"] = output

            entries_to_remove = ('description', 'test_list', 'code', 'full', 'input', 'output')
            for k in entries_to_remove:
                item.pop(k, None)

            if len(item['sample_actual_output'])>500:
                item.pop('sample_actual_output', None)

            if i < len(self.results):
                self.results.results[i] = item
                self.results.save_results()
            else:
                self.results.add_result(item)

            if self.verbose:
                print(
                    f'completed {i+1}/{num_items}, '
                    f'Solved: {self.results[i]["is_solved"]},'
                    f'Solved Sample: {any([x>=0.999 for x in self.results[i]["is_solved_sample"]])}'
                )

            # break
        # output the results to an output folder
        output_results(self.results.result_path[:-1], self.data)

    def generate_scorer(self, item):
        # for approximation questions we need to have a scorer

        cwd = os.path.dirname(os.path.abspath(__file__))
        prompts_scorer = utils.load_prompts(os.path.join(cwd, 'prompt_templates/prompts_scorer.yaml'))
        scorer_prompt = prompts_scorer["scorer"].format(problem_prompt=self.data.get_prompt(item))

        scorer = ""
        for _ in range(3):
            code_output, _, _ = self.model.prompt(
                processed_input=[{"role": "user", "content": scorer_prompt}],
                temperature=0.1
            )
            scorer = utils.parse_code(code_output).strip()

            if scorer.startswith("lambda") and "pred" in scorer and "true" in scorer:
                try:
                    scorer_func = eval(scorer)
                    if scorer_func(1, 1):
                        self.data.scorer = scorer_func
                        break
                    else:
                        print("Scorer function does not return True for identical inputs.")
                except Exception as e:
                    print(f"Error evaluating scorer: {e}")

        # check that we are actually dealing with an approximation problem, otherwise we don't need a scorer
        if "<" not in scorer and ">" not in scorer:
            self.data.scorer = None
        if self.data.scorer is not None:
            print(f"Problem identified as allowing for approximate outputs, using scorer function: \n {scorer}")


