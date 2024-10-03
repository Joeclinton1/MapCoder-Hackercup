from mapcoder_hackercup.models.ModelFactory import ModelFactory
import os
import promptings.utils as utils
import json
import importlib
import sys
from tqdm import tqdm
import argparse  # Import argparse for command line argument parsing
import re
import faiss
import numpy as np

# We have a module called datasets already, this causes a problem
# Temporarily remove the current directory from sys.path, import module, then restore original path
current_path = sys.path.pop(0)
import datasets
from sentence_transformers import SentenceTransformer

sys.path.insert(0, current_path)


def save_tags_to_json(problem_name, tags, output_file):
    # Load existing data from the JSON file
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    # Add or update the entry for the current problem
    data[problem_name] = tags

    # Write the updated data back to the file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def process_buffer(model, buffer):
    if not buffer:
        return
    prompts_list, names_list = zip(*buffer)

    def gen_tags_inner(i):
        response, _, _ = model.prompt(processed_input=[{"role": "user", "content": prompts_list[i]}])
        return i, response  # Return the index along with the response

    print(f"Processing buffer with {len(buffer)} prompts...")
    # Run in parallel and collect results with their indices
    indexed_tags_list = utils.run_func_parallel_and_collect(gen_tags_inner, num_parallel=NUM_PARALLEL)

    # Sort results based on the original indices to maintain order
    indexed_tags_list.sort(key=lambda x: x[0])
    tags_list = [tags for _, tags in indexed_tags_list]

    for name, tags in zip(names_list, tags_list):
        print(f"Saving tags for problem: {name}")
        save_tags_to_json(name, tags, output_file)


# Function to extract tags from dataset entries
def extract_tags(prompts_file, gpu='4090', num_parallel=4):
    prompts = utils.load_prompts(prompts_file)
    dataset = datasets.load_dataset("deepmind/code_contests", split='test', streaming=True)
    dataset_length = dataset.info.splits['test'].num_examples

    model = ModelFactory.get_model_class("Codestral" if gpu == '4090' else "CodestralVLLM")(
        temperature=0.1,
        top_p=0.9,
        gpu=gpu
    )
    model.model_params.update(num_ctx=2048)

    buffer = []

    for entry in tqdm(dataset, desc="Generating Tags", total=dataset_length):
        problem_name = entry['name']
        problem_description = entry['description']
        prompt = prompts['tag_gen'].format(problem_prompt=problem_description)
        buffer.append((prompt, problem_name))

        if len(buffer) >= num_parallel:
            process_buffer(model, buffer)
            buffer = []

    if buffer:
        print("Processing remaining entries in buffer...")
        process_buffer(model, buffer)

    print("Tag generation completed.")


def post_process_output(output_file):
    if not os.path.exists(output_file):
        print(f"Output file '{output_file}' does not exist.")
        return

    with open(output_file, 'r') as f:
        data = json.load(f)

    processed_data = {}

    # Regex patterns to handle different formats
    regex_1 = re.compile(r'[-*]?\s*(\d*\.\d+)\s*[:()]\s*(\w+)', re.IGNORECASE)
    regex_2 = re.compile(r'Weight:\s*(\d*\.\d+)\s*,\s*Tag:\s*(\w+)', re.IGNORECASE)
    regex_3 = re.compile(r'\(Weight:\s*(\d*\.\d+)\)\s*[:()]\s*(\w+)', re.IGNORECASE)

    def extract_tags(tag_string):
        # Attempt each regex in sequence until a match is found
        for regex in [regex_1, regex_2, regex_3]:
            matches = regex.findall(tag_string)
            if matches:
                return [(float(weight), tag) for weight, tag in matches]
        return []

    def post_process_tags(tags):
        # Replace underscores with spaces
        processed_tags = [(weight, tag.replace('_', ' ')) for weight, tag in tags]
        # Filter out tags with "competitive coding" or "competitive programming"
        processed_tags = [(weight, tag) for weight, tag in processed_tags if
                          "competitive coding" not in tag.lower() and "competitive programming" not in tag.lower()]
        return processed_tags

    def rescale_weights(tags):
        # Rescale weights to sum to 1
        total_weight = sum(weight for weight, _ in tags)
        if total_weight != 0:
            return [(weight / total_weight, tag) for weight, tag in tags]
        return tags

    for problem_name, tags in data.items():
        extracted_tags = extract_tags(tags)
        # Post-process and rescale weights
        processed_tags = post_process_tags(extracted_tags)
        processed_tags = rescale_weights(processed_tags)
        processed_data[problem_name] = processed_tags

    # Save the processed data back to the JSON file (or to a new file if needed)
    processed_output_file = output_file.replace('.json', '_processed.json')
    with open(processed_output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print(f"Post-processing completed. Processed data saved to '{processed_output_file}'.")


def tags_to_faiss_index(output_file, model_name='all-MiniLM-L6-v2', index_file='data/code_contests.index'):
    if not os.path.exists(output_file):
        print(f"Output file '{output_file}' does not exist.")
        return

    # Load the JSON data from the input file
    with open(output_file, 'r') as f:
        data = json.load(f)

    model = SentenceTransformer(model_name)
    all_embeddings = []
    all_weights = []
    tags_list = []
    problem_names = []
    problem_embeddings = {}

    # Iterate through each problem in the JSON file
    for problem_key, problem_data in data.items():
        tags = problem_data
        weights = [tag[0] for tag in tags]
        tag_texts = [tag[1] for tag in tags]

        # Generate embeddings for each tag
        embeddings = model.encode(tag_texts)

        # Store embeddings, weights, and tags
        all_embeddings.extend(embeddings)
        all_weights.extend(weights)
        tags_list.extend(tag_texts)

        # Store embeddings for each problem to calculate the weighted average
        problem_embeddings[problem_key] = (embeddings, weights)

    # Convert all embeddings to a NumPy array (needed for FAISS)
    embeddings_array = np.array(all_embeddings).astype('float32')

    # Create a FAISS index (using L2 distance)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the FAISS index
    index.add(embeddings_array)

    # Save individual embeddings to a TSV file for visualization
    embeddings_tsv_file = output_file.replace('.json', '_embeddings.tsv')
    with open(embeddings_tsv_file, 'w') as f:
        for embedding in embeddings_array:
            f.write('\t'.join(map(str, embedding)) + '\n')

    # Save tag metadata to a TSV file
    metadata_tsv_file = output_file.replace('.json', '_metadata.tsv')
    with open(metadata_tsv_file, 'w') as f:
        f.write("Weight\tTag\n")
        for weight, tag in zip(all_weights, tags_list):
            f.write(f"{weight}\t{tag}\n")

    # Calculate weighted average embeddings for each problem and save to a separate TSV file
    avg_embeddings_tsv_file = output_file.replace('.json', '_avg_embeddings.tsv')
    with open(avg_embeddings_tsv_file, 'w') as f:
        for problem_key, (embeddings, weights) in problem_embeddings.items():
            # Calculate the weighted average embedding
            weights = np.array(weights)
            weights_sum = np.sum(weights)
            weighted_avg_embedding = np.average(embeddings, axis=0, weights=weights / weights_sum)

            # Save the weighted average embedding to TSV
            f.write('\t'.join(map(str, weighted_avg_embedding)) + '\n')

    # Save problem name metadata for the weighted average embeddings (no header)
    avg_metadata_tsv_file = output_file.replace('.json', '_avg_metadata.tsv')
    with open(avg_metadata_tsv_file, 'w') as f:
        for problem_key in problem_embeddings.keys():
            f.write(f"{problem_key}\n")

    print(f"FAISS index has been created and saved to '{index_file}'.")
    print(f"Embeddings have been saved to '{embeddings_tsv_file}'.")
    print(f"Metadata (tags) has been saved to '{metadata_tsv_file}'.")
    print(f"Weighted average embeddings have been saved to '{avg_embeddings_tsv_file}'.")
    print(f"Metadata (problem names) for weighted averages has been saved to '{avg_metadata_tsv_file}'.")


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Generate tags, post-process the output JSON file, or convert tags to embeddings.')
    parser.add_argument('--post_process', action='store_true',
                        help='If set, post-process the output JSON to extract (weight, tag) tuples.')
    parser.add_argument('--convert_to_embeddings', action='store_true',
                        help='If set, convert tags to embeddings.')
    parser.add_argument('--tags_only', action='store_true',
                        help='If set, extract tags only.')

    args = parser.parse_args()

    # Path to the prompts YAML file and output JSON file
    cwd = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(cwd, 'promptings/prompt_templates/prompts_tag_gen.yaml')
    output_file = os.path.join(cwd, '../../outputs/output_tags.json')
    GPU = '4090'
    NUM_PARALLEL = 10

    # If post-process flag is set, perform post-processing
    if args.post_process:
        post_process_output(output_file)

    # If convert_to_embeddings flag is set, perform embedding conversion
    elif args.convert_to_embeddings:
        processed_file = output_file.replace('.json', '_processed.json')
        tags_to_faiss_index(processed_file)

    # If tags_only flag is set, extract tags only
    elif args.tags_only:
        extract_tags(prompts_file, gpu=GPU, num_parallel=NUM_PARALLEL)

    # If no flags are set, perform all three tasks in sequence
    else:
        # Step 1: Post-process the output JSON to extract (weight, tag) tuples
        post_process_output(output_file)

        # Step 2: Convert tags to embeddings
        processed_file = output_file.replace('.json', '_processed.json')
        tags_to_faiss_index(processed_file)

        # Step 3: Extract tags
        extract_tags(prompts_file, gpu=GPU, num_parallel=NUM_PARALLEL)
