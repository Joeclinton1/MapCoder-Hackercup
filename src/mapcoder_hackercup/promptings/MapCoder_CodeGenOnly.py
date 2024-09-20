import os

from .MapCoder import MapCoder
from . import utils

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_custom.yaml')


class CodeGenOnly(MapCoder):
    def __init__(self, k: int = 3, t: int = 5, *args, **kwargs):
        super().__init__(k, t, *args, **kwargs)
        self.prompts = utils.load_prompts(prompts_file)

    def run_single_pass(self, item: dict):
        print("", flush=True)

        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\
        \nWe will use a breadth-first search (BFS) algorithm to efficiently explore connected components on a 2D grid.\
        \nThe BFS ensures that we can check all possible white stone clusters for capturing, and we will also keep track of adjacent empty spaces to verify capture conditions.\
        \nThe algorithm updates a dynamic programming table to record the maximum number of stones captured at each position, backtracking where necessary."

        plan = f"\
        1. Read the input dimensions R (rows) and C (columns), and store the board in a 2D array A.\
        \n2. Initialize a lookup array to track visited cells and a dp array to store the maximum stones captured at each position.\
        \n3. For each white stone ('W') on the board that hasn't been visited yet, perform a breadth-first search (BFS) to explore all connected white stones.\
        \n4. During BFS, also track adjacent empty spaces ('adj') to determine whether the group of white stones can be captured.\
        \n5. If the white stone group has exactly one adjacent empty space, mark that space and update the dp array to record the number of stones captured by placing a black stone there.\
        \n6. Continue this process for all white stones, and keep track of the maximum number of white stones captured for any move.\
        \n7. Output the maximum number of stones captured for each test case."

        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"
        plannings = [[plan,None,None]]
        code, pr_tok, com_tok = self.generate_final_code(item, plannings, algorithm_prompt, sample_io_prompt, 0, 0)

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok