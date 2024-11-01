import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import time

from google.protobuf import descriptor as _descriptor
import ray

import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from huggingface_hub import snapshot_download


def create_captions(
    df,
    batch_size,
    model_name="llama3-instruct-molinst-molecule-8b",
    adapter_name="zjunlp/llama3-instruct-molinst-molecule-8b",
):

    name = model_name.split("/")[-1]

    if adapter_name is not None:
        lora_path = snapshot_download(adapter_name)

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    def build_input(selfies_input):
        instruction = "Please give me some details about this molecule:"
        input = selfies_input
        input = f'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"'

        return input

    df["input"] = df["SELFIES"].progress_map(build_input)

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=512, min_p=0.15, top_p=0.85
    )

    lq = LoRARequest(adapter_name, 1, lora_path) if adapter_name is not None else None

    def get_captions(SMIs):
        all_caps = []

        # for smis in tqdm(divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs)/batch_size)):
        prompts = SMIs

        caps = llm.generate(prompts, sampling_params, lora_request=lq)

        caps = [c.outputs[0].text for c in caps]
        # caps = [c.replace("<|start_header_id|>assistant<|end_header_id|>", '').strip() for c in caps]

        all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["input"].tolist())

    df["captions"] = captions

    return df, name
