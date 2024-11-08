# PromptMonkey: Multi-Agent Many Shot Code Generation for Meta Hackercup Problem Solving

As part of the Meta Hackercup Ai Track competition (for more info see [https://hackercupai.github.io/](here)), we rewrote and extended the [https://github.com/Md-Ashraful-Pramanik/MapCoder](MapCoder) LLM agent to be effective for meta hackercup problem solving.

Our overall strategy was:

- **Scaffolding & Pipeline**: Careful prompt engineering was essential to getting the small LLM to produce high quality code/reasoning that could be successfully parsed and executed.
- **Observations & COT**: We generated a pool of observations about the problem and step by step reasoning that we randomly selected from to concatenate to the problem statement.
- **Codestral-22B**: we tested many models and found this was the best base model that fit comfortably in 40GB.
- **Maj@128**: We generated 128 code samples, tested them against the sample cases, and applied majority voting if multiple passed.
- **Inference speed**: Using VLLM for parallel inference and carefully tuned parameters we reached an average output of 2000 tk/s allowing for more tokens per question without exceeding the strict time limit.
- **Code improvement**: Repeatedly improve the best scoring samples, until they passed.

We tested various strategies which you can see in our strategies folder. Our best strategy was our many shot one which we have labelled as baseline.

## Setup Our Project
1. Clone our project
```
git clone https://github.com/Joeclinton1/MapCoder-Hackercup && cd MapCoder-Hackercup
```

2. Install the module using the following command:
```
pip install -e ./
```

3. Set up the .env file by seeing the example.


4. Generate the dataset by running: 
```
python src/mapcoder_hackercup/datasets/convert-hackercup-xcode.py
```

5. Start the local llm model in a Windows terminal with Ollama installed 
```
set OLLAMA_HOST=0.0.0.0
Ollama serve
```
6. setup the [ExecEval](https://github.com/ntunlp/ExecEval) for docker execution. Please visit this [link](https://github.com/ntunlp/ExecEval) to setup a docker container and run it using 5000 port. Change the line 50 of the file `src\evaluations\api_comm.py` for different setup.
```
docker run -it -p 5000:5000 -e NUM_WORKERS=23 exec-eval:1.0
```

## Running Our Project

1. To run MapCoder with Codestral on the sample dataset run
```
python -m mapcoder_hackercup --model Codestral
```
2. To run MapCoder with Codestral on the full dataset run
```
python -m mapcoder_hackercup --model Codestral --split Full
```
3. To run map coder with the first available Ollama model on the sample dataset run
```
python -m mapcoder_hackercup --model Local
```

4. To run map coder with codestral on just the ready_go_part_2 problem of the sample dataset
```
python -m mapcoder_hackercup --model Codestral --problem_ids ready_go_part_2
```

5. To run map coder with Local model on all problems with a different temperature and top p each stage
```
python -m mapcoder_hackercup --model Local --temperature 0.7 0.6 0 --top_p 0.7 0.8 1
```

6. Running one problem from practice round on Baseline strategy and ideal settings
```
python -m mapcoder_hackercup --model Codestral --strategy Baseline --language Python3 --top_p 0.9 --dataset Live --dir contestData --problem_ids "Line of Delivery (Part 2)"
```

6. Running all problems from practice round on Baseline strategy and ideal settings, using VLLM on A100
```
python -m mapcoder_hackercup --model CodestralVLLM --strategy Baseline --language Python3 --top_p 0.9 --dataset Live --dir contestData
```

## Running LLMs on containers (e.g Vast.ai)

### Step 1
```
sudo apt update
sudo apt install -y pipx
pipx install vllm
pipx install hf_transfer
pipx install "huggingface_hub[cli]"
export HF_HUB_ENABLE_HF_TRANSFER=1
pipx ensurepath
tmux kill-server
```
After it installs it logs you out, so log back in again

### Step 2
We tested loads of models here are some options
( note qwen2.5 models came out after the cutoff date set in the rules so they're only here for comparison):

- Option 1 - Codestral-22b - INT4 - AWQ 
  - AWQ (recommended)
    ```
    vllm serve solidrust/Codestral-22B-v0.1-hf-AWQ   --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0 --chat-template "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.last and system_message is defined %}\n            {{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}\n        {%- else %}\n            {{- '[INST] ' + message['content'] + '[/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n" 
    ```
  - GPTQ
    ```
    vllm serve ArthurGprog/Codestral-22B-v0.1-FIM-Fix-GPTQ --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0 --chat-template "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.last and system_message is defined %}\n            {{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}\n        {%- else %}\n            {{- '[INST] ' + message['content'] + '[/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"
    ```

- Option 2: Qwen 2.5 14B 
  - Int4
    ```
    vllm serve Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0 
    ```

  - Int 8
    ```
    vllm serve Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8 --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0 
    ```

- Option 3: Qwen2.5 34B - INT4
```
 vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0
```

- Option 4: Llama 3.1 70B - INT4

```
-- 40GB VRAM version
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16 --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0 --gpu-memory-utilization 1.0 --max-model-len 4096 --enforce_eager --cpu-offload-gb 1.5 --max_num_seqs 2

-- 80GB VRAM version
vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16 --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0 --max-model-len 6144 --max_num_seqs 32
```

- Option 5: Qwen2.5 70B - INT4
```
 vllm serve Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 --dtype auto --api-key token-abc123 --port 11434 --host 0.0.0.0 --max-model-len 6144 --max_num_seqs 32  
```




## Citation
```
@article{islam2024mapcoder,
  title={MapCoder: Multi-Agent Code Generation for Competitive Problem Solving},
  author={Islam, Md Ashraful and Ali, Mohammed Eunus and Parvez, Md Rizwan},
  journal={arXiv preprint arXiv:2405.11403},
  year={2024}
}
```
