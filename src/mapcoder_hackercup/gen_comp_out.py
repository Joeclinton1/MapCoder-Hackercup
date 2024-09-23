import os
import json
from mapcoder_hackercup.datasets.HackercupDataset import HackercupDataset
from mapcoder_hackercup.datasets.Live import LiveDataset

# Function to handle the outputting process
def output_results(results_path, dataset):
    # Load the results from the JSONL file
    with open(results_path, 'r') as file:
        data = json.load(file)

    # Extract RUN_NAME from the results path
    run_name = os.path.splitext(os.path.basename(results_path))[0]

    # Create the output directory
    output_dir = os.path.join(os.path.dirname(results_path), f"out_{run_name}")
    os.makedirs(output_dir, exist_ok=True)
    dataset_task_to_idx = {item[dataset.id_key]: idx for idx, item in enumerate(dataset)}

    # Iterate through each problem in the results
    for problem in data:
        task_id = problem["task_id"]
        source_code = problem["source_codes"][0]  # Take the 0th index of source_codes
        if "full_output" in problem and problem["is_solved"] != "error" :
            output = problem["full_output"]
        else:
            full_input = dataset.data[dataset_task_to_idx[task_id]]['full']
            item = {
                'test_list': [dict(input=full_input, output=[""])],
                dataset.id_key: task_id
            }
            # Generate output using ExecEval
            is_solved, output = dataset.evaluate(
                item=item,
                cur_imp=source_code,
                language=problem["language"]
            )

        # Write source code to a file
        source_code_filename = os.path.join(output_dir, f"{task_id}_source_code.py")
        with open(source_code_filename, 'w') as f:
            f.write(source_code)

        # Write output to a file
        try:
            output_filename = os.path.join(output_dir, f"{task_id}_output.txt")
            with open(output_filename, 'w') as f:
                f.write(output)
        except:
            pass

    print(f"Output written to folder: {output_dir}")


# Main runner function for executing the script
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True
    )

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

    args = parser.parse_args()

    match args.dataset:
        case 'Hackercup':
            dataset = HackercupDataset(split=args.split)
        case 'Live':
            if not args.dir:
                raise ValueError(f'Please specify the dir name!')
            dataset = LiveDataset(dir_name=args.dir[0])
        case _:
            raise ValueError(f'Please specify a valid dataset!')
    output_results(args.path, dataset)