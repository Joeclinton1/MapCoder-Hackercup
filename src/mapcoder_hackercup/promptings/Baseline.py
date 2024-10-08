'''
This strategy is all about staying simple in the structure, and just going ham with samples.
Simple but effective with small models
'''
import os
from .Base import BaseStrategy
from . import utils
from ..results import write_debug
from random import choice, choices

cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_baseline.yaml')
lang_specific_file = os.path.join(cwd, 'prompt_templates/lang_specific_tips.yaml')

# constants
NUM_PARALLEL = 128


class Baseline(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super(Baseline, self).__init__(*args, **kwargs)
        self.pr_tok, self.com_tok = 0, 0
        self.sample_io_prompt = None
        self.prompts = utils.load_prompts(prompts_file)
        tips = utils.load_prompts(lang_specific_file)
        self.lang_specific_tips = f"# Language specific tips:\n{tips[self.language]}" if self.language in tips else ""

    def run_single_pass(self, item):
        print(f"Solving problem: {item['name']}")
        self.sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"
        problem = self.data.get_prompt(item)

        print(f"Generating a pool of 32 observations about the problem")
        obs = utils.run_func_parallel_and_collect(lambda i: self.generate_observation(item, problem), 32)
        print(f"Generating {NUM_PARALLEL} codes")
        results = utils.run_func_parallel_and_collect(
            lambda i: self.generate_code(item, problem, choices(obs,k=3)), NUM_PARALLEL
        )
        best_res = None
        for i in range(2):
            results.sort(key=lambda x: x[0], reverse=True)
            best_res = results[0]
            print(f' Scores: {",".join([str(r[0]) for r in results])}')
            print(f' Best Score: {best_res[0]}\n')

            if best_res[0] == 1:
                passed_codes = [x[1] for x in results if x[0] == 1]

                if len(passed_codes)<NUM_PARALLEL//8:
                    # take the solutions that passed and randomly sample to use as seeds for improvements
                    # robustify!

                    print(f"Generating {NUM_PARALLEL} more solutions from the solutions that passed")
                    results2 = utils.run_func_parallel_and_collect(
                        lambda i: self.generate_code_improvement(item, problem, choice(passed_codes), type="A"),
                        NUM_PARALLEL
                    )

                    print(f' Additional Scores: {",".join([str(r[0]) for r in results2])}')

                    results.extend(results2)

                passed = [x for x in results if x[0] == 1]
                mode_output_idx, count = utils.plurarity_vote([x[2] for x in passed])
                code = passed[mode_output_idx][1]
                print(f"Solution was voted {count}/{len(passed)} times")
                return code, self.pr_tok, self.com_tok
            elif i == 0:
                break
                # print("No solution passed, so lets try and fix the best ones we got.")
                # best_codes = [x[1] for x in results if x[0] == best_res[0]]
                # results2 = utils.run_func_parallel_and_collect(
                #     lambda i: self.generate_code_improvement(item, problem, choice(best_codes), type="B"),
                #     NUM_PARALLEL
                # )
                # print(f' Additional Scores: {",".join([str(r[0]) for r in results2])}')
                # results.extend(results2)

        code = best_res[1]
        return code, self.pr_tok, self.com_tok

    def generate_observation(self, item, problem_prompt):
        observation_prompt = self.prompts['observation'].format(
            problem_prompt=problem_prompt,
            sample_io_prompt=self.sample_io_prompt,
        )
        observation = self.chat(observation_prompt, item, tag='observation', temperature=1.0)
        return observation

    def generate_code(self, item, problem_prompt, obs):
        obs = [x for x in obs if len(x)<100]
        observations = '\n'.join(obs)
        code_prompt = self.prompts['coding'].format(
            problem_prompt=problem_prompt,
            language=self.language,
            lang_specific_tips=self.lang_specific_tips,
            sample_io_prompt=self.sample_io_prompt,
            observations=observations
        )
        code_output = self.chat(code_prompt, item, tag='code', temperature=1.0)
        code = utils.parse_code(code_output)
        score, test_result = self.data.evaluate_sample_io(item, code, self.language, log_if_passed_samples=True)
        return score, code, test_result

    def generate_code_improvement(self, item, problem_prompt, code, type):
        code_prompt = self.prompts[f'coding_improvement_{type}'].format(
            problem_prompt=problem_prompt,
            language=self.language,
            lang_specific_tips=self.lang_specific_tips,
            sample_io_prompt=self.sample_io_prompt,
            code=code
        )
        code_output = self.chat(code_prompt, item, tag='improvement', temperature=1.0)
        code = utils.parse_code(code_output)
        score, test_result = self.data.evaluate_sample_io(item, code, self.language, log_if_passed_samples=True)
        return score, code, test_result

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
