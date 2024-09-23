import os, pathlib
import zipfile
import shutil

from .Dataset import Dataset

from mapcoder_hackercup.evaluations.evalute import contest_evaluate, contest_evaluate_public_tests
from mapcoder_hackercup.constants.paths import LIVE_DATA_DIR


class LiveDataset(Dataset):
    def __init__(self, dir_name, problem_ids=None, password=None):
        self.dir_name = dir_name
        self.password = password
        super().__init__(os.path.join(LIVE_DATA_DIR, dir_name), problem_ids=problem_ids, id_key="name")

    def load(self):
        data = []
        for problem in os.listdir(self.path):
            problem_path = pathlib.Path(self.path) / problem

            problem_data = dict()
            for file_name, key in zip(['statement.txt', 'sample_in.txt', 'sample_out.txt', 'full_in.txt'],
                                      ['description',   'input',         'output',         'full']):
                with open(problem_path / file_name, 'r') as f:
                    problem_data[key] = f.read()
            problem_data['sample_io'] = [dict(input=problem_data['input'], output=[problem_data['output']])]
            problem_data['sample_io'] = [dict(input=problem_data['input'], output=[problem_data['output']])]
            problem_data['test_list'] = [dict(input=problem_data['full'], output=[""])]
            problem_data['name'] = problem
            data.append(problem_data)

        if self.problem_ids is not None:
            data = [item for item in data if item[self.id_key] in self.problem_ids]
        self.data = data

    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        return contest_evaluate(
            generated_code=cur_imp,
            id=item[self.id_key],
            tests=item["test_list"],
            lang=language
        )

    def evaluate_sample_io(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        return contest_evaluate_public_tests(
            generated_code=cur_imp,
            id=item[self.id_key],
            tests=[dict(input=item['input'], output=[item['output']])],
            lang=language
        )

    @staticmethod
    def get_prompt(item):
        return f"""{item['description']}\n
        Sample Input:\n{item['input']}
        Sample Output:\n{item['output']}\n
        Important: Follow the input/output format strictly. Read input from standard input and write output to standard output.
        """
