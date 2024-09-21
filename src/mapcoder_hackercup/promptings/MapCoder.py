import os
import time

from . import utils
from .Base import BaseStrategy
from ..results import write_debug
from typing import List

# Path to the prompts YAML file
cwd = os.path.dirname(os.path.abspath(__file__))
prompts_file = os.path.join(cwd, 'prompt_templates/prompts_mapcoder.yaml')

class MapCoder(BaseStrategy):
    def __init__(
            self,
            k: int = 3,
            t: int = 5,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.t = t
        self.prompts = utils.load_prompts(prompts_file)
        self.pr_tok = 0
        self.com_tok = 0

    def run_single_pass(self, item: dict):
        print("", flush=True)

        # Step 1: Generate KB exemplars and algorithm
        response = self.generate_kb_exemplars_and_algorithm(item)

        # Step 2: Generate plannings based on examples and sort by confidence
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        plannings = self.generate_plannings(item, response, algorithm_prompt, sample_io_prompt)
        plannings = [x[0] for x in sorted(plannings, key=lambda x: x[1], reverse=True)]

        # Step 3: For each planning generate code. Iteratively improve code until it passes samples cases.
        code = self.generate_final_code(item, plannings, algorithm_prompt, sample_io_prompt)

        print("________________________\n\n", flush=True)
        return code, self.pr_tok, self.com_tok

    def generate_kb_exemplars_and_algorithm(self, item):
        """Create input for knowledge base and exemplars using YAML template."""
        kb_exemplar_input = self.prompts['kb_exemplars']['content'].format(
            problem_prompt=self.data.get_prompt(item),
            mapping_k=self.k,
            language=self.language
        )

        utils.log("Input for knowledge base and exemplars: ", kb_exemplar_input)
        response = self.chat(kb_exemplar_input, item)
        response = self.post_process_response(response)
        utils.log("Response from knowledge base and exemplars: ", response)
        response = utils.parse_xml(response)

        return response

    def post_process_response(self, response):
        """Trim and format the response XML."""
        response = utils.trim_text(response, "# Identify the algorithm")
        response = utils.trim_text(response, "# Write a useful tutorial")
        response = utils.trim_text(response, "# Planning to solve this problem:")
        response = utils.trim_text(response, f"# Let's think step by step to solve this problem in {self.language}")
        response = utils.replace_tag(response, 'algorithm')
        response = utils.replace_tag(response, 'description')
        response = utils.replace_tag(response, 'code')
        response = utils.replace_tag(response, 'planning')
        return response

    def generate_plannings(self, item, response, algorithm_prompt, sample_io_prompt):
        """Generate planning using exemplars and verify confidence."""
        plannings = []
        for example_no, example in enumerate(response["problem"], start=1):
            example_problem = example["description"]
            example_planning = example["planning"]

            write_debug(dict(
                no=example_no, description=example_problem, plan=example_planning), 'exemplar')

            input_for_problem_planning = self.prompts['problem_planning_input']['content'].format(
                example_problem=example_problem,
                example_planning=example_planning,
                algorithm_prompt=algorithm_prompt,
                problem_prompt=self.data.get_prompt(item),
                sample_io_prompt=sample_io_prompt
            )

            utils.log("Input for our problem planning using example {example_no}:", input_for_problem_planning)
            planning = self.chat(input_for_problem_planning, item)

            utils.log(f"Response from our problem planning (example {example_no}): ",planning)

            verification_res = self.verify_planning(item, planning)
            utils.log("Response from planning verification (example {example_no}): ", verification_res)
            write_debug(dict(confidence=verification_res['confidence'], plan=planning), 'planning')

            plannings.append((planning, verification_res['confidence'], example))

        return plannings

    def verify_planning(self, item, planning):
        """Verify if the generated planning is correct and obtain confidence."""
        input_for_verification = self.prompts['verification_input']['content'].format(
            language=self.language,
            problem_prompt=self.data.get_prompt(item),
            planning=planning
        )

        utils.log("Input for planning verification: ", input_for_verification)
        verification_res, _, _ = self.chat(input_for_verification, item)

        verification_res = utils.parse_xml(utils.replace_tag(verification_res, 'confidence'))
        return verification_res

    def generate_final_code(self, item, plannings, algorithm_prompt, sample_io_prompt):
        """Generate and improve code until all test cases pass."""

        best_score, best_code = 0.0, ""
        for planning in plannings:
            input_for_final_code_generation = self.prompts['code_generation_input']['content'].format(
                language=self.language,
                algorithm_prompt=algorithm_prompt,
                problem_prompt=self.data.get_prompt(item),
                planning=planning,
                sample_io_prompt=sample_io_prompt,
                std_input_prompt=self.prompts['std_input_prompt']['content']
            )

            utils.log("Input for final code generation:", input_for_final_code_generation)
            code = self.chat(input_for_final_code_generation, item)
            code = utils.parse_code(code)
            utils.log("Response from final code generation: ", code)

            score, code = self.run_sample_tests(item, code, algorithm_prompt)

            if score > best_score:
                best_score, best_code = score, code

            if score == 1.0:
                write_debug(code, 'code')
                return code

        write_debug(best_code, 'code')
        return best_code

    def run_sample_tests(self, item, code, algorithm_prompt):
        """Run the sample test cases on the generated code."""
        best_score, best_code = 0.0, ""

        for i in range(1, self.t + 1):
            score, test_log = self.data.evaluate_sample_io(item, code, self.language)
            write_debug(dict(score=score, feedback=test_log, code=code), 'improvement')

            if score > best_score:
                best_score, best_code = score, code

            if score == 1.0:
                break
            print(f"Test case failed. Attempt {i} - test log: ")
            print(test_log, flush=True)

            code = self.improve_code(item, code, test_log, algorithm_prompt)

        return best_score, best_code

    def improve_code(self, item, code, test_log, algorithm_prompt):
        """Improve the generated code based on test case failures."""
        input_for_code_improvement = self.prompts['code_improvement_input']['content'].format(
            language=self.language,
            algorithm_prompt=algorithm_prompt,
            problem_prompt=self.data.get_prompt(item),
            test_log=test_log,
            response=code,
            std_input_prompt=self.prompts['std_input_prompt']['content']
        )

        utils.log("Input for improving code generation: ", input_for_code_improvement)
        response = self.chat(input_for_code_improvement, item)
        code = utils.parse_code(response)
        utils.log("Response from improving code generation: ", response)

        return code

    def chat(self, input: str, item: dict, **kwargs) -> (str, int, int):
        item['api_calls'] = item.get('api_calls', 0) + 1
        response, pr_tok, com_tok = self.model.prompt(
            processed_input= [{"role": "user","content": input}],
            **kwargs
        )
        self.pr_tok += pr_tok
        self.com_tok += com_tok
        return response