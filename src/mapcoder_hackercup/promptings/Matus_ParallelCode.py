import os

from .Matus import Matus
from . import utils
import concurrent.futures

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_matus.yaml')


class ParallelCode(Matus):
    def __init__(self, *args, **kwargs):
        self.prompts = utils.load_prompts(prompts_file)
        self.n_plans = kwargs.get('n_plans', 7)

        self.pr_tok, self.com_tok = 0, 0

        super(Matus, self).__init__(*args, **kwargs)

    def generate_code(self, item, plan, max_score, max_code, problem_prompt, sample_io_prompt):
        # Function to generate code and evaluate it
        def process_code():
            # Create a code prompt using the same plan each time
            code_prompt = self.prompts['coding']['content'] \
                .format(problem_prompt=problem_prompt,
                        plan=plan,
                        sample_io_prompt=sample_io_prompt,
                        std_input_prompt=self.prompts['std_input_prompt']['content'],
                        language=self.language)

            # Generate the code using the chat model
            code_output = self.chat(code_prompt, item, tag='code')

            # Parse the generated code
            code = utils.parse_code(code_output)

            # Evaluate the code and return the score and test result
            score, test_result = self.data.evaluate_sample_io(item, code, self.language)
            return score, code

        # Running the code generation in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit the same process 4 times
            futures = [executor.submit(process_code) for _ in range(4)]

            # Wait for all futures to complete and gather results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Select the result with the highest score
        best_score, best_code = max(results, key=lambda x: x[0])
        scores = ",".join([str(r[0]) for r in results])
        print(f' Scores: {scores}')
        print(f' Best Score: {best_score}')

        return best_score, best_code