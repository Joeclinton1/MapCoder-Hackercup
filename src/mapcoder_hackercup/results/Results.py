import os
from pathlib import Path
import shutil
import datetime as dt

from mapcoder_hackercup.utils.jsonl import read_jsonl, write_jsonl, write_json

"""
In this file, we define the Results class, 
which is used to store the results of the simulation.

It will take a result path at first and after each 
simulation, it will save the results in that path.

Results are in the form of a list of dictionaries
and will be saved as a jsonl file.
"""

debug_path = Path(f'{os.path.dirname(__file__)}/../../../debug').resolve()
if debug_path.exists():
    shutil.rmtree(debug_path)


debug_path.mkdir(parents=True)
def write_debug(dict_, type_):
    if isinstance(dict_, str):
        with open(debug_path / f'{dt.datetime.now().isoformat()}_{type_}.txt', 'w') as f:
            f.write(dict_)
    elif isinstance(dict_, dict):
        with open(debug_path / f'{dt.datetime.now().isoformat()}_{type_}.txt', 'w') as f:
            for key, val in dict_.items():
                f.write(key + ' ')
                f.write(str(val))
                f.write('\n\n')

class Results(object):
    def __init__(
        self, 
        result_path: str, 
        discard_previous_run: bool = False
    ):
        self.result_path = result_path

        self.discard_previous_run = discard_previous_run
        self.load_results()

    def add_result(self, result: dict):
        write_debug(result, 'result')

        self.results.append(result)
        self.save_results()

    def save_results(self):
        # write_jsonl(self.result_path, self.results)
        write_json(self.result_path[:-1], self.results)

    def load_results(self):
        if os.path.exists(self.result_path):
            if self.discard_previous_run:
                os.remove(self.result_path)
            else:
                self.results = read_jsonl(self.result_path)
        else:
            self.results = []

    def get_results(self):
        return self.results

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]
