"""
Keith (artcmd)
updated by 9/23/2025

Qwen3 with vLLM as offline inference engine
model: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
GPU: one single Nvidia H100

package version in conda environment: python=3.11.13
Name                       Version          Build            Channel
vllm                       0.10.2           pypi_0           pypi
torch                      2.8.0            pypi_0           pypi
tokenizers                 0.22.0           pypi_0           pypi
transformers               4.57.0.dev0      pypi_0           pypi
"""

import json
import os
from datetime import datetime
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def prepare_chat_messages(prompts: List[str], tokenizer) -> List[str]:
    formatted_prompts = []
    for prompt in prompts:
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        formatted_prompts.append(text)
    return formatted_prompts


def extract_response_from_output(output_text):
    """
    The thinking content should be between <think> and </think> tags

    Note: Qwen3 output contains only </think> without an explicit opening <think> tag
    """
    try:
        # Find the last </think> tag and extract content after it
        if '</think>' in output_text:
            thinking = output_text.split('</think>')[0].strip()
            content = output_text.split('</think>')[-1].strip()
        else:
            thinking = ''
            content = output_text.strip()
        return content, thinking
    except Exception:
        thinking = ''
        content = output_text.strip()
        return content, thinking


def write_prompt_thinking_response(output_path, prompt, thinking_chain, response):
    with open(output_path, 'w') as file:
        file.write(prompt)
        file.write('\n\n-------- Thinking --------\n')
        file.write(thinking_chain)
        file.write('\n\n-------- Response --------\n')
        file.write(response)
        file.flush()


def use_vllm(model_name='Qwen/Qwen3-30B-A3B-Thinking-2507-FP8',
             gpu_memory=0.9, context_len=65536,
             sp_max=32768, sp_temp=0.6, sp_p=0.95, sp_k=20, sp_rp=1.0, sp_pp=0.0):
    name = model_name.split('/')[-1]
    prompt_dir = '../llm_prompts'

    current_time = datetime.now().strftime('%y%m%d%H%M%S')
    output_dir = f'../output/{name}_{current_time}'
    thinking_dir = f'../thinking/{name}_{current_time}'
    tensor_parallel_size = torch.cuda.device_count()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(thinking_dir, exist_ok=True)
    print('[I:use_vllm]', tensor_parallel_size, 'GPUs are detected!')

    # https://docs.vllm.ai/en/v0.10.2/api/vllm/#vllm.LLM
    llm = LLM(model=model_name,
              dtype='auto',
              tensor_parallel_size=tensor_parallel_size,
              trust_remote_code=True,
              gpu_memory_utilization=gpu_memory,
              max_model_len=context_len)
    # https://docs.vllm.ai/en/v0.10.2/api/vllm/sampling_params.html
    sampling_params = SamplingParams(max_tokens=sp_max,
                                     temperature=sp_temp,
                                     top_p=sp_p,
                                     top_k=sp_k,
                                     repetition_penalty=sp_rp,
                                     presence_penalty=sp_pp,
                                     seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_prompts = []
    file_mappings = []
    prompt_files = [os.path.join(prompt_dir, file) for file in os.listdir(prompt_dir) if file.endswith('.json')]
    prompt_files = sorted(prompt_files)
    for prompt_file in prompt_files:
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)

        start_idx = len(all_prompts)
        all_prompts.extend(prompts)
        end_idx = len(all_prompts)

        file_mappings.append({'file': prompt_file,
                              'start_idx': start_idx,
                              'end_idx': end_idx
                              })
        print(f'{os.path.basename(prompt_file)} has {len(prompts)} prompts.')
    print(f'Total prompts to process: {len(all_prompts)}')

    prompt_texts = [item['llm_prompt'] for item in all_prompts]
    formatted_prompts = prepare_chat_messages(prompt_texts, tokenizer)
    all_outputs = llm.generate(formatted_prompts, sampling_params)

    mapping_dict = {}
    for mapping in file_mappings:
        name = os.path.basename(mapping['file']).split('_prompts')[0]
        start_idx = mapping['start_idx']
        end_idx = mapping['end_idx']
        for idx in range(start_idx, end_idx):
            mapping_dict[idx] = name

    all_responses = []
    for i, output in enumerate(tqdm(all_outputs, desc='Processing outputs')):
        generated_text = output.outputs[0].text
        response, thinking_chain = extract_response_from_output(generated_text)
        all_responses.append({'identifier': all_prompts[i]['identifier'],
                              'llm_output': response
                              })

        idx = all_prompts[i]['identifier']
        thinking_path = os.path.join(thinking_dir, f'{mapping_dict[i]}-{i:03d}-{idx}.txt')
        write_prompt_thinking_response(thinking_path, prompt_texts[i], thinking_chain, response)

    for mapping in file_mappings:
        file_responses = all_responses[mapping['start_idx']:mapping['end_idx']]

        outputs_save_path = os.path.join(output_dir,
                                         f"{os.path.basename(mapping['file']).split('_prompts')[0]}_outputs.json")
        os.makedirs(os.path.dirname(outputs_save_path), exist_ok=True)

        with open(outputs_save_path, 'w') as f:
            json.dump(file_responses, f, indent=4)
        print(f'Saved {len(file_responses)} responses to {outputs_save_path}')


def vllm_test():
    try:
        use_vllm(model_name='Qwen/Qwen3-30B-A3B-Thinking-2507-FP8',
                 context_len=65536,
                 gpu_memory=0.9,
                 sp_max=8192,
                 sp_temp=0.6,
                 sp_p=0.95,
                 sp_k=20,
                 sp_rp=1.00,
                 sp_pp=0.0)
    except Exception as e:
        print('[E]', e)


if __name__ == '__main__':
    vllm_test()
