import sys
from datetime import datetime
from mapcoder_hackercup.constants.paths import *

from mapcoder_hackercup.models.Gemini import Gemini
from mapcoder_hackercup.models.OpenAI import OpenAIModel

from mapcoder_hackercup.results.Results import Results

from mapcoder_hackercup.promptings.PromptingFactory import PromptingFactory
from mapcoder_hackercup.models.ModelFactory import ModelFactory

from mapcoder_hackercup.datasets.HackercupDataset import HackercupDataset
from mapcoder_hackercup.datasets.Live import LiveDataset

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--split",
    type=str, 
    default="Sample",
    choices=[
        "Sample",
        "Full",
    ]
)

parser.add_argument(
    "--strategy", 
    type=str, 
    default="MapCoder", 
    choices=[
        "Direct",
        "CoT",
        "MapCoder",
        "Custom",
        "DirectPlanning",
        "Matus",
        "ParallelCode",
        "Joe",
        "Zac",
        "Baseline"
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
        "Local",
        "Deepseek",
        "Llama",
        "CodestralVLLM",
        "LlamaVLLM",
        "QwenVLLM"
    ]
)

parser.add_argument(
    "--temperature", 
    type=float, 
    default=[0],
    nargs='+',
)

parser.add_argument(
    "--top_p",
    type=float,
    default=[0.95],
    nargs='+',
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

parser.add_argument(
    "--dataset",
    type=str,
    default='Hackercup',
    choices=[
        'Hackercup',
        'Live'
    ]
)

parser.add_argument(
    '--dir',
    type=str,
    default=None,
    nargs='+'
)

parser.add_argument(
    '--plan',
    type=str,
    default=None,
    help='If set prompt strategy will skip planning stage and use this plan instead'
)

parser.add_argument(
    '--improvement_dir',
    type=str,
    default=None,
    help='If set prompt strategy will skip straight to improving the code and plans at the provided directory.'
)

parser.add_argument(
    '--gpu',
    type=str,
    default='4090',
    choices=[
        '4090',
        'A100'
    ]
)

parser.add_argument(
    '--password',
    type=str,
    default=None,
)

args = parser.parse_args()

SPLIT = args.split
STRATEGY = args.strategy
MODEL_NAME = args.model
TEMPERATURE = args.temperature
TOP_P = args.top_p
PASS_AT_K = args.pass_at_k
LANGUAGE = args.language
PROBLEM_IDS = args.problem_ids
DATASET = SPLIT if args.dataset == 'Hackercup' else args.dataset

RUN_NAME = f"{MODEL_NAME}-{STRATEGY}-{DATASET}-{LANGUAGE}-{TEMPERATURE[0]}-{PASS_AT_K}"
if args.improvement_dir:
    RUN_NAME = f"{MODEL_NAME}-{STRATEGY}-{DATASET}-{LANGUAGE}-{TEMPERATURE[0]}-improvement"
os.makedirs('./outputs', exist_ok=True)
RESULTS_PATH = f"./outputs/{RUN_NAME}.jsonl"

print(f"#########################\nRunning start {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

match args.dataset:
    case 'Hackercup':
        dataset = HackercupDataset(problem_ids=PROBLEM_IDS, split=SPLIT)
    case 'Live':
        if not args.dir:
            raise ValueError(f'Please specify the dir name!')
        dataset = LiveDataset(problem_ids=PROBLEM_IDS, dir_name=args.dir[0], password=args.password)
    case _:
        raise ValueError(f'Please specify a valid dataset!')

strategy = PromptingFactory.get_prompting_class(STRATEGY)(
    model=ModelFactory.get_model_class(MODEL_NAME)(
        temperature=TEMPERATURE[0] if TEMPERATURE else None,
        top_p=TOP_P[0] if TOP_P else None,
        gpu=args.gpu
    ),
    data=dataset,
    language=LANGUAGE,
    pass_at_k=PASS_AT_K,
    results=Results(RESULTS_PATH),
    temps = TEMPERATURE,
    top_ps = TOP_P,
    plan = args.plan,
    improvement_dir = args.improvement_dir
)

strategy.run()

print(f"#########################\nRunning end {RUN_NAME}, Time: {datetime.now()}\n##########################\n")

