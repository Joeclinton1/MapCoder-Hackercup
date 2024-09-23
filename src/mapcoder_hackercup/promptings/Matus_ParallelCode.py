import os

from .Matus import Matus
from . import utils
import concurrent.futures

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_matus.yaml')
prompts_file_2 = os.path.join(cwd, 'prompt_templates/prompts_parallelcode.yaml')

NUM_PARALLEL = 4
IMPROVE_PLAN = False

utils.log = lambda a, b: None
class ParallelCode(Matus):
    def __init__(self, *args, **kwargs):
        self.prompts = utils.load_prompts(prompts_file)
        self.prompts2 = utils.load_prompts(prompts_file_2)
        # self.n_plans = kwargs.get('n_plans', 7)

        self.pr_tok, self.com_tok = 0, 0

        super(Matus, self).__init__(*args, **kwargs)

    def run_single_pass(self, item):
        def gen_plan():
            return self.chat(planning_prompt, item, 'breakdown')

        problem_prompt = self.data.get_prompt(item)
        planning_prompt = self.prompts['breakdown']['content']\
            .format(problem_prompt=problem_prompt)
        sample_io_prompt=f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}"

        max_score, max_code = 0.0, ""

        print(f' Generating {NUM_PARALLEL} plans ')
        plans = self.run_func_parallel_and_collect(gen_plan)

        for i, plan in enumerate(plans):
            print(f' --- Attempt {i} --- ')
            print('Generating code')
            score, code = self.generate_code(
                item, plan, max_score, max_code, problem_prompt, sample_io_prompt
            )
            if score == 1.0:
                break
            if score > max_score:
                max_score, max_code = score, code
        return max_code, self.pr_tok, self.com_tok
    def generate_code(self, item, plan, max_score, max_code, problem_prompt, sample_io_prompt):
        # Function to generate code and evaluate it
        test_log = None
        best_score, best_code = 0.0, ""
        def gen_initial_code():
            code_prompt = self.prompts['coding']['content'] \
                .format(problem_prompt=problem_prompt,
                        plan=plan,
                        sample_io_prompt=sample_io_prompt,
                        std_input_prompt=self.prompts['std_input_prompt']['content'],
                        language=self.language)

            code_output = self.chat(code_prompt, item, tag='code')
            code = utils.parse_code(code_output)
            score, test_result = self.data.evaluate_sample_io(item, code, self.language)
            return score, code, test_result

        def improve_code():
            """Improve the generated code based on test case failures."""
            input_for_code_improvement = self.prompts2['improve_plan_and_code']['content'].format(
                language=self.language,
                problem_prompt=self.data.get_prompt(item),
                planning=plan,
                test_log=test_log,
                code=best_code,
                std_input_prompt=self.prompts['std_input_prompt']['content'],
            )

            response = self.chat(input_for_code_improvement, item, tag='improve')
            code = utils.parse_code(response)
            score, test_result = self.data.evaluate_sample_io(item, code, self.language)
            return score, code, test_result
        funcs_labels = [(gen_initial_code, "Initial Code Generation"), (improve_code, "Improve Code")]
        for func, label in funcs_labels[:IMPROVE_PLAN+1]:
            results = self.run_func_parallel_and_collect(func)
            best_score, best_code, test_report = max(results, key=lambda x: x[0])

            scores = ",".join([str(r[0]) for r in results])
            print(f" {label}:")
            print(f' Scores: {scores}')
            print(f' Best Score: {best_score}\n')

        return best_score, best_code


    def run_func_parallel_and_collect(self, func):
        # Running the code generation in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL) as executor:
            futures = [executor.submit(func) for _ in range(NUM_PARALLEL)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return results