import os
import json

# Path to the directory containing the JSON files
directory = "outputs"

# Iterate over files in the directory
for filename in os.listdir(directory):
    # Check if the filename contains "improvement" and is a file
    file_path = os.path.join(directory, filename)
    if "improvement" in filename and os.path.isfile(file_path):
        # Extract the strategy name
        strategy_name = filename.split('-')[1]

        # Initialize lists to store scores and times for the current strategy
        is_solved_sample_scores = []
        times = []

        # Load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Iterate over the list of problem results in the JSON
            for result in data:
                # Check if the 'name' key is "Line of Delivery (Part 2)" and skip if true
                if result.get('name') == "Line of Delivery (Part 2)":
                    continue

                # Extract the is_solved_sample scores and flatten them
                is_solved_sample_scores.extend(result['is_solved_sample'])

                # Calculate the time in seconds using the given formula
                prompt_tokens = sum(result.get('prompt_tokens', []))
                completion_tokens = sum(result.get('completion_tokens', []))
                time_seconds = round(((prompt_tokens / 1500) + (completion_tokens / 400))/7,1)
                times.append(time_seconds)

        # Calculate the proportion of scores that are equal to 1
        proportion_solved = is_solved_sample_scores.count(1) / len(
            is_solved_sample_scores) if is_solved_sample_scores else 0

        # Calculate the average time
        average_time = sum(times) / len(times) if times else 0

        # Print the results in the specified format
        print(f"Strategy: {strategy_name}")
        print(f"Scores: {is_solved_sample_scores}")
        print(f"Proportion: {proportion_solved}")
        print(f"Times (in seconds): {times}")
        print(f"Average Time (in seconds): {average_time}\n")
