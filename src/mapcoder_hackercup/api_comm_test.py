import requests
from evaluations.api_comm import APICommunication, ExtendedUnittest

# Sample data
language = "Python 3"
source_code = """
def add(a, b):
    return a + b
"""
unittests = [
    {
        "input": "1, 2",
        "output": ["3"],
        "result": "passed"
    },
    {
        "input": "5, 7",
        "output": ["12"],
        "result": "passed"
    }
]

# Updated limits with only 'nofile' constraint
limits = {
    "nofile": 4  # Limit number of open files
}

try:
    # Create APICommunication instance and test the execute_code method
    with APICommunication(server_url="http://windows-6absj2b:5000") as api_comm:
        result, sample_id, task_id = api_comm.execute_code(
            language=language,
            source_code=source_code,
            unittests=unittests,
            limits=limits,  # Only 'nofile' limit
            block_network=True,
            stop_on_first_fail=True
        )

        print("Execution Result:", result)
        print("Sample ID:", sample_id)
        print("Task ID:", task_id)

except requests.exceptions.RequestException as e:
    print(f"Error communicating with API: {e}")
