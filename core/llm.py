import sys
import json
import time
from core.api_config import *
import transformers
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

default_decoding_args = {
    "max_new_tokens": 100,
    "do_sample": False,  # enable sampling
    "top_p": 0.9,  # nucleus sampling
    "temperature": 0.6,  # lower makes the distribution sharper
    "min_length": None,
    "use_cache": True,
    "top_k": 100,  # restrict to top-k probability tokens
    "repetition_penalty": 1.,  # 1 means no penalty; up to inf
    "length_penalty": 1.,  # length_penalty > 0.0 == longer sequences; length_penalty < 0.0 == shorter sequences
    "num_beams": 10,  # beam search
    "num_return_sequences": 10,  # number of beams to return
    "no_repeat_ngram_size": 10,
    "renormalize_logits": True
}

MAX_TRY = 5

world_dict = {}

log_path = None
api_trace_json_path = None
total_prompt_tokens = 0
total_response_tokens = 0


def api_func(prompt:str):
    global MODEL_NAME
    print(f"\nUse OpenAI model: {MODEL_NAME}\n")
    if 'mosaicml' in MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b-instruct", trust_remote_code=True)

        pipe = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.1,
            max_new_tokens=50,
            repetition_penalty=1.1
        )
        result = pipe(prompt)[0]["generated_text"]
        print(result)
    else:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
    text = response['choices'][0]['message']['content'].strip()
    prompt_token = response['usage']['prompt_tokens']
    response_token = response['usage']['completion_tokens']
    return text, prompt_token, response_token


def safe_call_llm(input_prompt, **kwargs) -> str:
    global MODEL_NAME
    global log_path
    global api_trace_json_path
    global total_prompt_tokens
    global total_response_tokens
    global world_dict

    for i in range(5):
        try:
            sys_response, prompt_token, response_token = api_func(input_prompt)
            return sys_response
        except Exception as ex:
            print(ex)
            print(f'Request {MODEL_NAME} failed. try {i} times. Sleep 20 secs.')
            time.sleep(20)

    raise ValueError('safe_call_llm error!')