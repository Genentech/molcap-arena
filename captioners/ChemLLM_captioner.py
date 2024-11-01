import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import time

# from llama3_chat import ChatSystem

from google.protobuf import descriptor as _descriptor
import ray

import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from huggingface_hub import snapshot_download

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def create_captions(
    df,  # batch_size=1,
    model_name="AI4Chem/ChemLLM-7B-Chat-1_5-DPO",
):
    batch_size = 1  # there's a bug in rotary embeddings with larger batch sizes
    name = model_name.split("/")[-1]

    # vllm is broken with rope scaling, see https://github.com/vllm-project/vllm/issues/4784
    if False:
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            enable_lora=True,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.9,
            max_new_tokens=500,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
        )

    if False:  # Results here seem worse..

        def InternLM2_format(instruction, prompt, answer, history):
            prefix_template = ["<|system|>:", "{}"]
            prompt_template = ["<|user|>:", "{}\n", "<|Bot|>:\n"]
            system = (
                f"{prefix_template[0]}\n{prefix_template[-1].format(instruction)}\n"
            )
            history = "\n".join(
                [
                    f"{prompt_template[0]}\n{prompt_template[1].format(qa[0])}{prompt_template[-1]}{qa[1]}"
                    for qa in history
                ]
            )
            prompt = f"\n{prompt_template[0]}\n{prompt_template[1].format(prompt)}{prompt_template[-1]}"
            return f"{system}{history}{prompt}"

        def build_input(smiles_input):
            sys_prompt = '''- Chepybara is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be Professional, Sophisticated, and Chemical-centric. 
    - For uncertain notions and data, Chepybara always assumes it with theoretical prediction and notices users then.
    - Chepybara can accept SMILES (Simplified Molecular Input Line Entry System) string, and prefer output IUPAC names (International Union of Pure and Applied Chemistry nomenclature of organic chemistry), depict reactions in SMARTS (SMILES arbitrary target specification) string. Self-Referencing Embedded Strings (SELFIES) are also accepted.
    - Chepybara always solves problems and thinks in step-by-step fashion, Output begin with *Let's think step by step*."'''

            instruction = "Could you provide a description of this molecule?"

            return InternLM2_format(instruction, sys_prompt, smiles_input, [])

    else:

        def build_input(smiles_input):
            instruction = "Could you provide a description of this molecule?"
            input = smiles_input
            input = f'{instruction}\n{input}\n\n"'

            return input

    df["input"] = df["SMILES"].progress_map(build_input)

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # sampling_params = SamplingParams(temperature=0.2, max_tokens=512, min_p=0.15, top_p=0.85)

    def get_captions(SMIs):
        all_caps = []

        for smis in tqdm(
            divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs) / batch_size)
        ):
            prompts = smis

            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")

            outputs = model.generate(
                **inputs, generation_config=generation_config
            ).squeeze()

            caps = [
                tokenizer.decode(
                    outputs[inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            ]

            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["input"].tolist())

    df["captions"] = captions

    return df, name
