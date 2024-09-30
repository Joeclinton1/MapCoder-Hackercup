from typing import List
import tiktoken
import os
import copy
import time

from mapcoder_hackercup.models.Base import BaseModel
from mapcoder_hackercup.datasets.Dataset import Dataset
from mapcoder_hackercup.results.Results import Results
from mapcoder_hackercup.utils.parse import parse_response
from mapcoder_hackercup.gen_comp_out import output_results


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
        code_dir: str = None
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
        self.code_dir = code_dir

    def gpt_chat(self, processed_input: List[dict], **kwargs) -> (str, int, int):
        return self.model.prompt(processed_input=processed_input, **kwargs)

    def run_single_pass(self, item: dict):
        pass

    def run_single_pass_no_planning(self, item: dict, plan: str):
        return self.run_single_pass(item)

    def run_single_pass_code_improvement_only(self, item: dict, code: str):
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

            while cur_pass < self.pass_at_k and not is_solved:
                for _ in range(10):
                    try:
                        if self.plan is not None:
                            response, prompt_tokens, completion_tokens = self.run_single_pass_no_planning(item, self.plan)
                        elif self.code_dir is not None:
                            result = self.run_single_pass_code_improvement_only(item, self.code_dir)
                            response, prompt_tokens, completion_tokens = result
                        else:
                            response, prompt_tokens, completion_tokens = self.run_single_pass(item)
                        break
                    except Exception as e:
                        print(f"Exception occured with error: {e}")
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
            item["full_output"] = output

            entries_to_remove = ('description', 'test_list', 'code', 'full', 'input', 'output')
            for k in entries_to_remove:
                item.pop(k, None)

            if i < len(self.results):
                self.results.results[i] = item
                self.results.save_results()
            else:
                self.results.add_result(item)

            if self.verbose:
                print(
                    f'completed {i+1}/{num_items}, '
                    f'Solved: {self.results[i]["is_solved"]},'
                    f'Solved Sample: {self.results[i]["is_solved_sample"][0]>=0.999}'
                )

            # break
        # output the results to an output folder
        output_results(self.results.result_path[:-1], self.data)
