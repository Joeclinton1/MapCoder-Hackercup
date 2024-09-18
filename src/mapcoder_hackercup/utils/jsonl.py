import json

# Read an jsonl file and convert it into a python list of dictionaries.
def read_jsonl(filename):
    """Reads a jsonl file and yields each line as a dictionary"""
    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            lines.append(json.loads(line))
    return lines

# Write a python list of dictionaries into a jsonl file
def write_jsonl(filename, lines):
    """Writes a python list of dictionaries into a jsonl file"""
    with open(filename, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line) + "\n")

    write_json(filename[:-1], lines)

def write_json(filename, lines):
    """Writes a python list of dictionaries into a regular JSON file"""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(lines, file, ensure_ascii=False, indent=4)
