import re
import xml.etree.ElementTree as ET
import yaml
import concurrent.futures

mapping = {
    1: "one (01)",
    2: "two (02)",
    3: "three (03)",
    4: "four (04)",
    5: "five (05)",
    6: "six (06)",
    7: "seven (07)",
    8: "eight (08)",
    9: "nine (09)",
}

def xml_to_dict(element):
    result = {}
    for child in element:
        if child:
            child_data = xml_to_dict(child)
            if child.tag in result:
                if isinstance(result[child.tag], list):
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = [result[child.tag], child_data]
            else:
                result[child.tag] = child_data
        else:
            result[child.tag] = child.text
    return result

def parse_xml_element(response: str) ->ET.Element:
    return parse_xml(response, use_dict=False)
def parse_xml(response: str, use_dict=True) -> dict:
    if '```xml' in response:
        response = response.replace('```xml', '')
    if '```' in response:
        response = response.replace('```', '')

    try:
        root = ET.fromstring(response)
    except:
        try:
            root = ET.fromstring('<root>\n' + response + '\n</root>')
        except:
            root = ET.fromstring('<root>\n' + response)
    return xml_to_dict(root) if use_dict else root


def parse_code(response: str) -> str:
    response = response.strip()
    if "```" not in response:
        return response

    code_pattern = r'```((.|\n)*?)```'
    if "```Python" in response:
        code_pattern = r'```Python((.|\n)*?)```'
    if "```Python3" in response:
        code_pattern = r'```Python3((.|\n)*?)```'
    if "```python" in response:
        code_pattern = r'```python((.|\n)*?)```'
    if "```python3" in response:
        code_pattern = r'```python3((.|\n)*?)```'
    if "```C" in response:
        code_pattern = r'```C((.|\n)*?)```'
    if "```c" in response:
        code_pattern = r'```c((.|\n)*?)```'
    if "```cpp" in response:
        code_pattern = r'```cpp((.|\n)*?)```'
    if "```C++" in response:
        code_pattern = r'```C\+\+((.|\n)*?)```'
    if "```c++" in response:
        code_pattern = r'```c\+\+((.|\n)*?)```'
    if "```Java" in response:
        code_pattern = r'```Java((.|\n)*?)```'
    if "```java" in response:
        code_pattern = r'```java((.|\n)*?)```'
    if "```Node" in response:
        code_pattern = r'```Node((.|\n)*?)```'
    if "```node" in response:
        code_pattern = r'```node((.|\n)*?)```'
    if "```Rust" in response:
        code_pattern = r'```Rust((.|\n)*?)```'
    if "```rust" in response:
        code_pattern = r'```rust((.|\n)*?)```'
    if "```PHP" in response:
        code_pattern = r'```PHP((.|\n)*?)```'
    if "```php" in response:
        code_pattern = r'```php((.|\n)*?)```'
    if "```Go" in response:
        code_pattern = r'```Go((.|\n)*?)```'
    if "```go" in response:
        code_pattern = r'```go((.|\n)*?)```'
    if "```Ruby" in response:
        code_pattern = r'```Ruby((.|\n)*?)```'
    if "```ruby" in response:
        code_pattern = r'```ruby((.|\n)*?)```'
    if "```C#" in response:
        code_pattern = r'```C#((.|\n)*?)```'
    if "```c#" in response:
        code_pattern = r'```c#((.|\n)*?)```'
    if "```csharp" in response:
        code_pattern = r'```csharp((.|\n)*?)```'

    code_blocks = re.findall(code_pattern, response, re.DOTALL)

    if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
        code_str = "\n".join(code_blocks[-1])
    elif type(code_blocks[-1]) == str:
        code_str = code_blocks[-1]
    else:
        code_str = response

    # Weird indentation error fix for python, count how many spaces there are, then strip
    n_spaces = len(code_str) - len(code_str.lstrip())
    code_str = code_str.replace('\n' + ' ' * n_spaces, '\n')

    return code_str.strip()


def trim_text(text: str, trimmed_text: str):
    return text.replace(trimmed_text, '').strip()


def replace_tag(text: str, tag: str):
    if f'<{tag}><![CDATA[' in text and f']]></{tag}>' in text:
        return text
    else:
        return text.replace(f'<{tag}>', f'<{tag}><![CDATA[').replace(f'</{tag}>', f']]></{tag}>').strip()


def get_sample_io_str(sample_io: any) -> str:
    if len(sample_io) > 0:
        if type(sample_io[0]) == str:
            return "\n".join(sample_io)
        if type(sample_io[0]) == dict:
            return "\n".join([f"Sample Input:\n{io['input']}\nSample Output:\n{io['output'][0]}" for io in sample_io])
    return sample_io

def load_prompts(prompts_file):
    """Load the YAML file containing the prompt templates."""
    with open(prompts_file, 'r') as file:
        return yaml.safe_load(file)


def log(header, content):
    print("\n\n________________________")
    print(header)
    print(content, flush=True)


def holistic_get_best_result(results):
    # Instead of max score being returned use the average of the top two scores.
    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_code, test_result = results[0]
    if best_score == 1.0 or len(results) == 1:
        return best_score, best_code, test_result

    # if the best score == 0, then it might be 0 meaning error or 0.0 meaning just wrong answer
    # a wrong answer solution is better than error so we will remove the 0 solutions
    if best_score == 0:
        results2 = [x for x in results if not (isinstance(x[0], int) and x[0] == 0)]
        if len(results2) == 0:
            results2 = results
        best_score, best_code, test_result = results2[0]

    # weighted holistic scoring so that the second-best score is taken into account
    # intuition is that if the second-best score is low but top is high then the plan is not actually good
    average_top_two_score = results[0][0]*0.6+results[1][0]*0.4
    return average_top_two_score, best_code, test_result


def run_func_parallel_and_collect( func, num_parallel):
    # Running the code generation in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(func, i) for i in range(num_parallel)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results