import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import os

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # for Gemma2

import subprocess


def create_captions(
    df,
    batch_size,
    prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    model_name="google/gemma-2-9b-it",
    adapter_name=None,
):

    name = model_name.split("/")[-1]

    if "27b" in name:
        print("GPUs:", torch.cuda.device_count())
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.95,
            max_model_len=1024,  # enable_lora=True,
            # dtype='float16', #bfloat16 going OOM for mysterious reasons on VLLM
            # tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            # max_num_seqs=1,
        )
    else:
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.95,
            max_model_len=1024,  # enable_lora=True,
            enforce_eager=True,
        )

    tokenizer = llm.get_tokenizer()

    def get_history(smi):

        history = [
            # {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt + "\nMolecule: {}\n".format(smi)},
        ]
        return history

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=512, min_p=0.15, top_p=0.85
    )

    subprocess.run("nvidia-smi")

    def get_captions(SMIs):
        all_caps = []

        # for smis in tqdm(divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs)/batch_size)):
        prompts = tokenizer.apply_chat_template(
            [get_history(smi) for smi in SMIs],
            tokenize=False,
        )

        caps = llm.generate(
            prompts,
            sampling_params,
            lora_request=(
                LoRARequest(adapter_name, 1, adapter_name)
                if adapter_name is not None
                else None
            ),
        )
        caps = [c.outputs[0].text for c in caps]

        all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["BRICS"].tolist())

    df["captions"] = captions

    return df, name
