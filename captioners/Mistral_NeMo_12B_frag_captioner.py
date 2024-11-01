import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import torch

import sys

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import pipeline
from datasets import Dataset


def create_captions(
    df,
    batch_size,
    sys_prompt="You are a helpful expert in medicinal chemistry. You are sharing your knowledge of known chemistry with a colleague.",
    prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    model_name="mistralai/Mistral-Nemo-Instruct-2407",
    adapter_name=None,
):

    name = model_name.split("/")[-1]

    if True:  # not supported by vllm, build from source crashes
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            max_model_len=1024,  # enable_lora=True,
        )

        tokenizer = llm.get_tokenizer()

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

    # chatbot = pipeline("text-generation", model=model_name, max_new_tokens=512, device='cuda')
    # chatbot.tokenizer.pad_token_id = chatbot.model.config.eos_token_id
    # chatbot.tokenizer.padding_side='left'

    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=512, min_p=0.15, top_p=0.85
    )

    def get_captions(SMIs):
        all_caps = []

        # for smis in tqdm(divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs)/batch_size)):
        hists = [get_history(smi) for smi in SMIs]
        # prompts = [chatbot.tokenizer.apply_chat_template(s, tokenize=False) for s in hists]
        prompts = tokenizer.apply_chat_template(
            hists,
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

        # data = Dataset.from_list(prompts)

        # for cap, p in tqdm(zip(chatbot(prompts, batch_size=batch_size, pad_token_id=chatbot.tokenizer.eos_token_id), prompts), total=len(prompts)):
        # for cap in tqdm(chatbot(prompts, batch_size=batch_size, pad_token_id=chatbot.tokenizer.eos_token_id)):
        # sys.stdout.flush()

        # cap = cap[0]['generated_text'].split('[/INST]')[-1]#[2]['content']
        # print(cap)
        # caps = chatbot(prompts, batch_size=batch_size, pad_token_id=chatbot.tokenizer.eos_token_id)
        # print(caps)
        # caps = [c[0]['generated_text'].split('[/INST]')[-1] for c in caps]
        # print(caps)
        caps = [c.outputs[0].text for c in caps]

        all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["SMILES"].tolist())

    df["captions"] = captions

    return df, name
