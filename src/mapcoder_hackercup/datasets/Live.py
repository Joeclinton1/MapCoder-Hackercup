import os, pathlib
import subprocess

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

        # Try to unzip the file if it is not a directory
        folder_path = pathlib.Path(self.path)
        if not folder_path.is_dir():
            # If not a directory, check if a .zip file exists
            zip_file_path = folder_path.with_suffix('.zip')
            if zip_file_path.exists():
                # Create a new folder with the same name as the zip file (without extension)
                output_dir = folder_path.with_suffix('')
                output_dir.mkdir(exist_ok=True)

                try:
                    # Unzip using 7z with a password, extracting to the newly created folder
                    subprocess.run(
                        ['7z', 'x', f'-p{self.password}', str(zip_file_path), f'-o{str(output_dir)}'],
                        check=True
                    )
                    # Update folder_path to point to the newly created directory
                except subprocess.CalledProcessError:
                    print(f"Failed to unzip {zip_file_path}.")
                    return  # Exit since we can't continue without the unzipped files

        for problem in os.listdir(self.path):
            problem_path = pathlib.Path(self.path) / problem

            # Check if the current path is a directory
            if not problem_path.is_dir():
                continue

            problem_data = dict()
            for file_name, key in zip(['statement.txt', 'sample_in.txt', 'sample_out.txt', 'full_in.txt'],
                                      ['description', 'input', 'output', 'full']):
                with open(problem_path / file_name, 'r') as f:
                    problem_data[key] = f.read()

            # Check if 'full_out.txt' exists and use it if it does
            full_out_path = problem_path / 'full_out.txt'
            if full_out_path.exists():
                with open(full_out_path, 'r') as f:
                    full_output = f.read()
            else:
                full_output = ""

            problem_data['sample_io'] = [dict(input=problem_data['input'], output=[problem_data['output']])]
            problem_data['test_list'] = [dict(input=problem_data['full'], output=[full_output])]
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
        evaluate_on_full_if_passed=True,
        log_if_passed_samples = False
    ):
        results = contest_evaluate_public_tests(
            generated_code=cur_imp,
            id=item[self.id_key],
            tests=[dict(input=item['input'], output=[item['output']])],
            lang=language
        )

        # evaluate
        if results[0] == 1 and evaluate_on_full_if_passed:
            if log_if_passed_samples:
                print("Passed sample input.")

            # Note we overwrite results on purpose so that we have the full outputs for plurarity voting
            results = contest_evaluate(
                generated_code=cur_imp,
                id=item[self.id_key],
                tests=item["test_list"],
                lang=language
            )

            if not (isinstance(results[0], float) or results[0] is True):
                return 0.999, f"Program passes Sample Cases, but fails on full input with error: `{results[1]}`," \
                              f" Error Type: {results[0]}"
            return 1.0, results[1]
        return  results

    @staticmethod
    def get_prompt(item):
        return item['description']
