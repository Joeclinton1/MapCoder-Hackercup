import json
import os
import numpy as np
import tqdm
from yaml import safe_load
from typing import List

from mapcoder_hackercup.constants.paths import DATA_DIR
from .api_comm import APICommunication
from .exec_outcome import ExecOutcome
from mapcoder_hackercup.constants.lang_mappings import LANGUAGE_MAPPING
from ..results import write_debug
from ..promptings.utils import round_floats_in_str

limits_by_lang_cfg_file = f"{os.path.dirname(__file__)}/limits_by_lang.yaml"

assert os.path.exists(
    limits_by_lang_cfg_file), "Need resource limit defaults for all runtimes, provide the path to default 'limits_by_lang.yaml' or to the modified one."

with open(limits_by_lang_cfg_file) as limit_cfg_rp:
    limits_by_lang = safe_load(limit_cfg_rp)

unittest_file = f"{DATA_DIR}/xCodeEval/unittest_db.json"
print(unittest_file)
assert os.path.exists(unittest_file), "Unittest file not found."

with open(unittest_file) as ut_rp:
    unittest_db = json.load(ut_rp)

api_comm = APICommunication(server_url=os.getenv('XCODE_SERVER_URL', 'http://windows-6absj2b:5000'))
FULL_TEST_TIME = 40


def score_output_cases(output, expected_output, scorer=None):
    custom_scorer = True
    if scorer is None:
        custom_scorer = False
        scorer = lambda pred, true: pred == true

    output = output.rstrip()
    expected_output = expected_output.rstrip()
    len_out = len(output.split('\n'))
    len_eout = len(expected_output.split('\n'))
    if len_out != len_eout:
        print(f"len output: {len_out}, len expected output: {len_eout}\n")
        return 0.0

    output, expected_output = round_floats_in_str(output, 6), round_floats_in_str(expected_output, 6)
    passed = 0
    failed = 0
    for i, (x, y) in enumerate(zip(output.split('\n'), expected_output.split('\n'))):
        if custom_scorer:
            x = x.split(": ", 1)[1].strip()
            y = y.split(": ", 1)[1].strip()
        if scorer(x, y):
            passed += 1
        else:
            failed += 1
    return passed / (passed + failed)


def generate_output(
        generated_code: str,
        lang: str,
        id: int,
        stdin: str,
):
    assert lang in LANGUAGE_MAPPING, f"language must be inside the supported language list: {LANGUAGE_MAPPING.keys()}"

    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=[dict(input=stdin, output='')],
        limits=limits_by_lang[LANGUAGE_MAPPING[lang]],
        task_id=id
    )

    if results == "error":
        return "error"

    return '\n'.join(results['output'])


def contest_evaluate(
        generated_code: str,
        lang: str,
        id: int,
        tests: List[dict],
        scorer=None
):
    assert lang in LANGUAGE_MAPPING, f"language must be inside the supported language list: {LANGUAGE_MAPPING.keys()}"

    limits = limits_by_lang[LANGUAGE_MAPPING[lang]]
    limits["cpu"] = FULL_TEST_TIME
    limits["_as"] = -1
    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=tests,
        limits=limits,
        task_id=id,
    )
    write_debug(results[0], type_="output_full")

    if results == "error":
        return "error", ""

    if results[0]['exec_outcome'] == ExecOutcome.PASSED.value:
        return True, results[0]['result']
    elif results[0]['exec_outcome'] == ExecOutcome.WRONG_ANSWER.value:
        return score_output_cases(results[0]['result'], tests[0]["output"][0], scorer), results[0]['result']
    return results[0]['exec_outcome'], results[0]['result']


def contest_evaluate_public_tests(
        generated_code: str,
        lang: str,
        id: int,
        tests: List[dict],
        scorer=None
):
    limits = limits_by_lang[LANGUAGE_MAPPING[lang]]
    limits["cpu"] = 1.5

    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=tests,
        limits=limits,
        task_id=id,
        stop_on_first_fail=False
    )

    input = tests[0]['input']
    expected_output = tests[0]['output'][0]
    output = results[0]['result']
    write_debug(results[0], type_="output")
    if results[0]['exec_outcome'] == ExecOutcome.PASSED.value:
        return 1, "All tests passed!"
    elif results[0]['exec_outcome'] == ExecOutcome.WRONG_ANSWER.value:
        score = score_output_cases(output, expected_output, scorer)
        feedback = f"Wrong Solution.\n Input:\n{input}\nExpected Output:\n{expected_output}\nYour output:\n{output}"
        return score, feedback
    return 0, f"Program Failed with error: `{output}`, Error Type: {results[0]['exec_outcome']}"
