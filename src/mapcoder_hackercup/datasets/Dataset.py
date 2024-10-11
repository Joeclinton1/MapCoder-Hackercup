from mapcoder_hackercup.utils.jsonl import read_jsonl


class Dataset(object):
    def __init__(
        self,
        path: str,
        problem_ids: list = None,
        id_key: str = ""
    ):
        self.path = path
        self.data = None
        self.id_key = id_key
        self.problem_ids = problem_ids
        self.scorer = None
        self.load()

    def load(self):
        self.data = read_jsonl(self.path)
        if self.problem_ids is not None:
            self.data = [item for item in self.data if item[self.id_key] in self.problem_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        raise NotImplementedError

    @staticmethod
    def get_prompt(item):
        raise NotImplementedError
