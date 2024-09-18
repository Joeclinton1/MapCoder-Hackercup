import os
from . import utils
from .Base import BaseStrategy

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

    def run_single_pass(self, item: dict):
        print("", flush=True)
        input_kb_exemplars = self.create_kb_exemplars(item)
        response, pr_tok, com_tok = self.gpt_chat(input_kb_exemplars)
        item['api_calls'] = item.get('api_calls', 0) + 1

        response = self.post_process_response(response)
        print("\n\n________________________")
        print("Response from knowledge base and exemplars: ")
        print(response, flush=True)

        response = utils.parse_xml(response)
        algorithm_prompt = f"## Relevant Algorithm to solve the next problem:\n{response['algorithm']}"
        sample_io_prompt = f"## Sample Test cases: \n{utils.get_sample_io_str(item['sample_io'])}\n"

        # Generate plannings based on examples
        plannings, pr_tok, com_tok = self.generate_plannings(item, response, algorithm_prompt, sample_io_prompt, pr_tok,
                                                             com_tok)

        # Sort plannings by confidence and generate code
        plannings.sort(key=lambda x: x[1], reverse=True)
        code, pr_tok, com_tok = self.generate_final_code(item, plannings, algorithm_prompt, sample_io_prompt, pr_tok,
                                                         com_tok)

        print("________________________\n\n", flush=True)
        return code, pr_tok, com_tok

    def create_kb_exemplars(self, item):
        """Create input for knowledge base and exemplars using YAML template."""
        kb_exemplar_template = self.prompts['kb_exemplars']['content']
        return [
            {
                "role": "user",
                "content": kb_exemplar_template.format(
                    problem=self.data.get_prompt(item),
                    k=utils.mapping[self.k],
                    language=self.language
                )
            },
        ]

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

    def generate_plannings(self, item, response, algorithm_prompt, sample_io_prompt, pr_tok, com_tok):
        """Generate planning using exemplars and verify confidence."""
        plannings = []
        for example_no, example in enumerate(response["problem"], start=1):
            example_problem = example["description"]
            example_planning = example["planning"]

            # Generate problem planning input using YAML template
            input_for_problem_planning = [
                {
                    "role": "user",
                    "content": self.prompts['problem_planning_input']['content'].format(
                        example_problem=example_problem,
                        example_planning=example_planning,
                        algorithm_prompt=algorithm_prompt,
                        problem_prompt=self.data.get_prompt(item),
                        sample_io_prompt=sample_io_prompt
                    )
                }
            ]

            planning, pr_tok_1, com_tok_1 = self.gpt_chat(input_for_problem_planning)
            item['api_calls'] += 1
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            verification_res = self.verify_planning(item, planning)
            plannings.append((planning, verification_res['confidence'], example))

        return plannings, pr_tok, com_tok

    def verify_planning(self, item, planning):
        """Verify if the generated planning is correct and obtain confidence."""
        input_for_verification = [
            {
                "role": "user",
                "content": self.prompts['verification_input']['content'].format(
                    language=self.language,
                    problem_prompt=self.data.get_prompt(item),
                    planning=planning
                )
            }
        ]
        verification_res, _, _ = self.gpt_chat(input_for_verification)
        item['api_calls'] += 1
        verification_res = utils.parse_xml(utils.replace_tag(verification_res, 'confidence'))
        return verification_res

    def generate_final_code(self, item, plannings, algorithm_prompt, sample_io_prompt, pr_tok, com_tok):
        """Generate and improve code until all test cases pass."""
        std_input_prompt = "## Note: Strictly follow input/output format using `input()` for input and print for output."
        for planning, confidence, example in plannings:

            # Generate code generation input using YAML template
            input_for_final_code_generation = [
                {
                    "role": "user",
                    "content": self.prompts['code_generation_input']['content'].format(
                        language=self.language,
                        algorithm_prompt=algorithm_prompt,
                        problem_prompt=self.data.get_prompt(item),
                        planning=planning,
                        sample_io_prompt=sample_io_prompt,
                        std_input_prompt=std_input_prompt
                    )
                }
            ]

            code, pr_tok_1, com_tok_1 = self.gpt_chat(input_for_final_code_generation)
            item['api_calls'] += 1
            code = utils.parse_code(code)
            pr_tok += pr_tok_1
            com_tok += com_tok_1

            passed = self.run_sample_tests(item, code)
            if passed:
                return code, pr_tok, com_tok
        return None, pr_tok, com_tok

    def run_sample_tests(self, item, code):
        """Run the sample test cases on the generated code."""
        passed = False
        for i in range(1, self.t + 1):
            passed, test_log = self.data.evaluate_sample_io(item, code, self.language)
            if passed:
                break
            code = self.improve_code(item, code, test_log)
        return passed

    def improve_code(self, item, code, test_log):
        """Improve the generated code based on test case failures."""
        input_for_code_improvement = [
            {
                "role": "user",
                "content": self.prompts['code_improvement_input']['content'].format(
                    language=self.language,
                    test_log=test_log,
                    code=code
                )
            }
        ]
        response, _, _ = self.gpt_chat(input_for_code_improvement)
        item['api_calls'] += 1
        return utils.parse_code(response)
