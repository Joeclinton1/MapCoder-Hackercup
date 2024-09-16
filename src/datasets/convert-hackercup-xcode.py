import json
from datasets import load_dataset
from tqdm import tqdm  # Import tqdm for progress bars

def write_jsonl(filename, lines):
    """Writes a list of dictionaries to a jsonl file."""
    with open(filename, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")

def get_test_cases(input, output):
    return {
        "input": "\n".join([str(x) for x in input]) if type(input) == list else input,
        "output": output if type(output) == list else [output]
    }

def process_dataset(split_type, output_filename):
    """Process and save the dataset based on the split type."""
    # Load the dataset from Hugging Face
    print(f"Loading {split_type} split of the dataset from Hugging Face...")
    dataset = load_dataset("hackercupai/hackercup", split=split_type)

    processed_data = []
    count_errors = 0

    # Iterate through the dataset rows with a progress bar
    print(f"Processing {split_type} split...")
    for item in tqdm(dataset, desc="Processing items", unit="item"):
        if any([x is None for x in item.values()]):
            count_errors += 1
            continue

        # Process sample test cases
        sample_io = list(map(get_test_cases, item['sample_input'], item['sample_output']))

        # Process full test cases (now renamed to test_list)
        test_list = {
            "input": item['input'],
            "solution": item['solution'],
            "output": item['output']
        }

        processed_item = {
            "name": item['name'],
            "year": item['year'],
            "round": item['round'],
            "description": item['statement'],
            "code": item['code'],
            "sample_io": sample_io,
            "test_list": test_list,
        }

        processed_data.append(processed_item)

    print(f"{count_errors} items skipped from the {split_type} split")

    # Write the processed data to a new JSONL file
    print(f"Writing {split_type} split to JSONL file...")
    write_jsonl(output_filename, processed_data)
    print(f"Data writing complete for {split_type} split.")

# Process and save the full split
process_dataset("full", "./data/Hackercup/hackercup_processed.jsonl")

# Process and save the sample split
process_dataset("sample", "./data/Hackercup/hackercup_processed_sample.jsonl")