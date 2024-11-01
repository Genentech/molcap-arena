import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import subprocess


def create_captions(
    df,
    batch_size,
    sys_prompt="You are a helpful expert in medicinal chemistry. You are sharing your knowledge of known chemistry with a colleague.",
    prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    adapter_name=None,
):

    name = model_name.split("/")[-1]

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.95,
        max_model_len=1024,  # enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
    )

    tokenizer = llm.get_tokenizer()

    subprocess.run(["nvidia-smi"])

    def get_history(smi):

        history = [
            {"role": "system", "content": sys_prompt},
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
        caps = [
            c.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
            for c in caps
        ]

        all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["SMILES"].tolist())

    df["captions"] = captions

    return df, name
