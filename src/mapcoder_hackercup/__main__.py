import sys
from datetime import datetime
from mapcoder_hackercup.constants.paths import *

from mapcoder_hackercup.models.Gemini import Gemini
from mapcoder_hackercup.models.OpenAI import OpenAIModel

from mapcoder_hackercup.results.Results import Results

from mapcoder_hackercup.promptings.PromptingFactory import PromptingFactory
from mapcoder_hackercup.datasets.DatasetFactory import DatasetFactory
from mapcoder_hackercup.models.ModelFactory import ModelFactory

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type=str, 
    default="HumanEval", 
    choices=[
        "HumanEval", 
        "MBPP", 
        "APPS",
        "xCodeEval", 
        "CC",
        "Hackercup",
        "HackercupSample"
    ]
)
parser.add_argument(
    "--strategy", 
    type=str, 
    default="MapCoder", 
    choices=[
        "Direct",
        "CoT",
        "SelfPlanning",
        "Analogical",
        "MapCoder",
        "Custom",
    ]
)
parser.add_argument(
    "--model", 
    type=str, 
    default="ChatGPT", 
    choices=[
        "ChatGPT",
        "GPT4",
        "Gemini",
        "Codestral",
        "Local"
    ]
)
parser.add_argument(
    "--temperature", 
    type=float, 
    default=0
)
parser.add_argument(
    "--pass_at_k", 
    type=int, 
    default=1
)
parser.add_argument(
    "--language", 
    type=str, 
    default="Python3", 
    choices=[
        "C",
        "C#",
        "C++",
        "Go",
        "PHP",
        "Python3",
        "Ruby",
        "Rust",
    ]
)

parser.add_argument(
    "--problem_ids",
    type=str,
    default=None,
    nargs='+',
    help='A list of problem ids to test from the dataset. If not included will test all problems.'
)

args = parser.parse_args()

DATASET = args.dataset
STRATEGY = args.strategy
MODEL_NAME = args.model
TEMPERATURE = args.temperature
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language
PROBLEM_IDS = args.problem_ids

RUN_NAME = f"{MODEL_NAME}-{STRATEGY}-{DATASET}-{LANGUAGE}-{TEMPERATURE}-{PASS_AT_K}"
os.makedirs('./outputs', exist_ok=True)
RESULTS_PATH = f"./outputs/{RUN_NAME}.jsonl"

print(f"#########################\nRunning start {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_NAME)(temperature=TEMPERATURE),
    data=DatasetFactory.get_dataset_class(DATASET)(problem_ids=PROBLEM_IDS),
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,
    results=Results(RESULTS_PATH),
)

strategy.run()

print(f"#########################\nRunning end {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

