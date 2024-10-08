'''
This strategy is all about staying simple in the structure, and just going ham with samples.
Simple but effective with small models
'''
import os
from mapcoder_hackercup.promptings.Base import BaseStrategy
from mapcoder_hackercup.promptings import utils
from mapcoder_hackercup.results import write_debug
from random import choice, choices, sample
import re
from tabulate import tabulate

cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, '../prompt_templates/prompts_baseline.yaml')
lang_specific_file = os.path.join(cwd, '../prompt_templates/lang_specific_tips.yaml')

# constants
NUM_PARALLEL = 128
NUM_TRICKS = 64
DEBUG = False


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

        # print(f"Generating a pool of 32 observations about the problem")
        # obs = utils.run_func_parallel_and_collect(lambda i: self.generate_observation(item, problem), 32)

        print(f"Generating a pool of {NUM_TRICKS} tricks for how to solve the problem")
        tricks = utils.run_func_parallel_and_collect(lambda i: self.generate_trick(item, problem, i), NUM_TRICKS)
        scored_tricks = {i: (trick, 0) for i,trick in tricks}
        if DEBUG:
            scored_tricks = self.score_tricks(item, problem, tricks)

        tricks = self.tricks_comparison_and_purge(item, problem, scored_tricks)

        print(f"Generating {NUM_PARALLEL} codes")
        results = utils.run_func_parallel_and_collect(
            lambda i: self.generate_code(item, problem, choice(tricks)), NUM_PARALLEL
        )
        best_res = None
        for i in range(2):
            results.sort(key=lambda x: x[0], reverse=True)
            best_res = results[0]
            print(f' Scores: {",".join([str(r[0]) for r in results])}')
            print(f' Best Score: {best_res[0]}\n')

            if best_res[0] == 1:
                passed_codes = [x[1] for x in results if x[0] == 1]

                if len(passed_codes) < NUM_PARALLEL // 8:
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

    def tricks_comparison_and_purge(self, item, problem, tricks):
        items = sample(list(tricks.values()), len(tricks))
        pairs = list(zip(items[::2], items[1::2]))

        def compare_tricks(i):
            trick_a, trick_b = pairs[i]
            comparison_prompt = self.prompts['trick_comparison'].format(
                problem_prompt=problem,
                trick_a=trick_a[0],
                trick_b=trick_b[0]
            )

            for _ in range(3):
                try:
                    better_trick_og = self.chat(comparison_prompt, item, tag=f'trick_comparison{i}', temperature=0.6)
                    better_trick_og = utils.replace_tag(better_trick_og, 'analysis')
                    better_trick_og = utils.replace_tag(better_trick_og, 'verdict')
                    better_trick = utils.parse_xml(better_trick_og)['verdict'].upper()
                    break
                except:
                    print("failed to parse trick comparison. Retrying ...")
            else:
                return i, 0, (trick_a[1], trick_b[1])

            better_trick_parsed = next((word for word in reversed(better_trick.split()) if word in ['A', 'B']), None)
            if better_trick_parsed is None:
                print(f"Didn't find verdict. Got: {better_trick}")
                # we couldn't get a trick, but we still need to output something here
                return i, 0, (trick_a[1], trick_b[1])

            better_idx = ["A", "B"].index(better_trick_parsed)
            # return 0 for trick A and 1 for trick B, plus also the trick scores
            return i, better_idx, (trick_a[1], trick_b[1])

        print("Pairing off tricks and judging which is better")
        judged_tricks = utils.run_func_parallel_and_collect(compare_tricks,num_parallel=len(pairs))
        passed_tricks = [pairs[i][better_idx] for i, better_idx, _ in judged_tricks]
        return passed_tricks
        # print("Judged pairs:")
        # for i, better_idx, (score_a, score_b) in judged_tricks:
        #     outcome = (better_idx == (score_b>score_a)) or score_b==score_a
        #     print(f"{i} : Choice = {better_idx} which is {outcome}: Scores = {score_a}, {score_b}")

    def score_tricks(self, item, problem, tricks):
        tricks = {idx: value for idx, value in tricks}
        print(f"Score tricks against ground truth for debugging purposes")
        trick_scores = sorted(utils.run_func_parallel_and_collect(
            lambda i: utils.score_answer(item, problem, tricks[i], self.chat, i), NUM_TRICKS
        ))
        data = [[idx for idx, _ in trick_scores], [f"{score}%" for _, score in trick_scores]]
        print(tabulate(data, tablefmt="plain"))
        # input()
        return {idx: (tricks[idx], score) for idx, score in trick_scores}

    def generate_trick(self, item, problem_prompt, i):
        observation_prompt = self.prompts['trick'].format(
            problem_prompt=problem_prompt,
            sample_io_prompt=self.sample_io_prompt,
        )
        observation = self.chat(observation_prompt, item, tag=f'trick_{i}', temperature=1.0)
        return i, observation

    def generate_observation(self, item, problem_prompt):
        observation_prompt = self.prompts['observation'].format(
            problem_prompt=problem_prompt,
            sample_io_prompt=self.sample_io_prompt,
        )
        observation = self.chat(observation_prompt, item, tag='observation', temperature=1.0)
        return observation

    def generate_code(self, item, problem_prompt, trick):
        # obs = [x for x in obs if len(x)<100]
        # observations = '\n'.join(obs)
        code_prompt = self.prompts['coding'].format(
            problem_prompt=problem_prompt,
            language=self.language,
            lang_specific_tips=self.lang_specific_tips,
            sample_io_prompt=self.sample_io_prompt,
            # observations=observations,
            trick=trick
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
