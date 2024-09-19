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


def xcode_evaluate(
    generated_code: str,
    src_uid: str,
    lang: str
):

    assert src_uid in unittest_db, "Can not find the task id or source id"

    assert lang in LANGUAGE_MAPPING, f"language must be inside the supported language list: {LANGUAGE_MAPPING.keys()}"

    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=unittest_db[src_uid],
        limits=limits_by_lang[LANGUAGE_MAPPING[lang]],
        task_id=src_uid,
    )

    if results == "error":
        return False

    passed = True
    for result in results:
        if result['exec_outcome'] != ExecOutcome.PASSED.value:
            passed = False
            break

    return passed


def xcode_execute_internal_test(
    generated_code: str,
    tests: List[dict],
    src_uid: str,
    lang: str
):
    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=tests,
        limits=limits_by_lang[LANGUAGE_MAPPING[lang]],
        task_id=src_uid,
        stop_on_first_fail=False
    )

    passed = True
    passed_feedback = []
    failed_feedback = []

    idx = 0
    try:
        for idx, result in enumerate(results):
            if result['exec_outcome'] == ExecOutcome.PASSED.value:
                passed_feedback.append(tests[idx])
            if result['exec_outcome'] != ExecOutcome.PASSED.value:
                failed_feedback.append(tests[idx])
                passed = False
    except:
        passed = False
        failed_feedback.extend(tests[idx:])

    feedback = f'Tested passed: \n{json.dumps(passed_feedback)}\n\nTests failed: \n{json.dumps(failed_feedback)}'

    return passed, feedback

def score_output_cases(output, expected_output):
    passed = 0
    failed = 0
    for i, (x, y) in enumerate(zip(output.split('\n'), expected_output.split('\n'))):
        if x == y:
            passed += 1
        else:
            failed += 1
    return passed / (passed + failed)
def contest_evaluate(
    generated_code: str,
    lang: str,
    id: int,
    tests: List[dict],
):
    assert lang in LANGUAGE_MAPPING, f"language must be inside the supported language list: {LANGUAGE_MAPPING.keys()}"

    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=tests,
        limits=limits_by_lang[LANGUAGE_MAPPING[lang]],
        task_id=id,
    )

    if results == "error":
        return "error" # False

    if results[0]['exec_outcome'] == ExecOutcome.PASSED.value:
        return True
    elif results[0]['exec_outcome'] == ExecOutcome.WRONG_ANSWER.value:
        return score_output_cases(results[0]['result'], tests[0]["output"][0])
    return False

def contest_evaluate_public_tests(
    generated_code: str,
    lang: str,
    id: int,
    tests: List[dict],
):
    results, _, _ = api_comm.execute_code(
        language=LANGUAGE_MAPPING[lang],
        source_code=generated_code,
        unittests=tests,
        limits=limits_by_lang[LANGUAGE_MAPPING[lang]],
        task_id=id,
        stop_on_first_fail=False
    )

    input = tests[0]['input']
    expected_output = tests[0]['output'][0]
    output = results[0]['result']

    if results[0]['exec_outcome'] == ExecOutcome.PASSED.value:
        return 1, "All tests passed!"
    elif results[0]['exec_outcome'] == ExecOutcome.WRONG_ANSWER.value:
        score = score_output_cases(output, expected_output)
        feedback = f"Wrong Solution.\n Input:\n{input}\nExpected Output:\n{expected_output}\nYour output:\n{output}"
        return score, feedback
    return 0, f"Program Failed with error: `{output}`, Error Type: {results[0]['exec_outcome']}"
