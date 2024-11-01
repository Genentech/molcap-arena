import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import torch

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


def create_captions(df, batch_size, model_name="OpenDFM/ChemDFM-13B-v1.0"):

    name = model_name.split("/")[-1]

    def build_input(smiles_input):
        input_text = "Can you please give detailed descriptions of the molecule below?\n{}".format(
            smiles_input
        )
        model_input = f"[Round 0]\nHuman: {input_text}\nAssistant:"

        return model_input

    df["input"] = df["SMILES"].progress_map(build_input)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    generation_config = GenerationConfig(
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.9,
        max_new_tokens=1024,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
    )

    def get_captions(inps):
        all_caps = []

        for prompts in tqdm(
            divide_chunks(inps, batch_size), total=math.ceil(len(inps) / batch_size)
        ):
            inputs = tokenizer(
                prompts, return_tensors="pt", truncation=True, padding=True
            ).to("cuda")

            outputs = model.generate(**inputs, generation_config=generation_config)
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            caps = [gt[len(it) :].strip() for gt, it in zip(generated_text, prompts)]

            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["input"].tolist())

    df["captions"] = captions

    return df, name
