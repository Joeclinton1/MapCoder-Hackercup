from .Dataset import Dataset
from mapcoder_hackercup.evaluations.evalute import contest_evaluate, contest_evaluate_public_tests
from mapcoder_hackercup.constants.paths import HACKERCUP_DATA_PATH, HACKERCUP_DATA_PATH_SAMPLE


class HackercupDataset(Dataset):
    def __init__(
        self,
        problem_ids: list = None,
        split: str = "Sample"
    ):
        path = HACKERCUP_DATA_PATH_SAMPLE if split == "Sample" else HACKERCUP_DATA_PATH
        super().__init__(path, problem_ids=problem_ids, id_key="name")

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
        evaluate_on_full_if_passed=True
    ):

        results = contest_evaluate_public_tests(
            generated_code=cur_imp,
            id=item[self.id_key],
            tests=item["sample_io"],
            lang=language,
        )

        # evaluate
        if results[0] == 1 and evaluate_on_full_if_passed:
            results2 = contest_evaluate(
                generated_code=cur_imp,
                id=item[self.id_key],
                tests=item["test_list"],
                lang=language
            )
            if not (isinstance(results2, float) or results2 is True):
                return 0.999, f"Program passes Sample Cases, but fails on full input with error: `{results2[1]}`," \
                              f" Error Type: {results2[0]}"
        return results

    @staticmethod
    def get_prompt(item):
        return item['description']