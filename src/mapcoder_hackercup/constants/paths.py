from os.path import join, dirname

DATA_DIR = join(dirname(__file__), "..", "..", "..", "data")

HACKERCUP_DATA_DIR =join(
    DATA_DIR,
    "Hackercup",
)

HACKERCUP_DATA_PATH =join(
    HACKERCUP_DATA_DIR,
    "hackercup_processed.jsonl",
)

HACKERCUP_DATA_PATH_SAMPLE =join(
    HACKERCUP_DATA_DIR,
    "hackercup_processed_sample.jsonl",
)

LIVE_DATA_DIR = join(
    DATA_DIR,
    "Live",
)
