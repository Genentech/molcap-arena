import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

from rdkit import Chem

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

import pandas as pd


def create_captions(df, batch_size):

    model_name = "GT4SD/multitask-text-and-chemistry-t5-base-augm"

    name = model_name.split("/")[-1]

    model_max_length = 512

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=model_max_length
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()

    def build_input(smiles_input):
        input = "Caption the following molecule: {}".format(smiles_input)

        return input

    df["input"] = df["SMILES"].progress_map(build_input)

    device = "cuda"

    model.to(device)

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def get_captions(SMIs):
        all_caps = []

        for smis in tqdm(
            divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs) / batch_size)
        ):
            batch = tokenizer(
                smis, return_tensors="pt", truncation=True, padding=True
            ).to(device)

            outputs = model.generate(**batch, num_beams=10, max_length=model_max_length)

            caps = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            caps = [output.split(tokenizer.eos_token)[0] for output in caps]
            caps = [output.replace(tokenizer.pad_token, "") for output in caps]
            caps = [output.strip() for output in caps]

            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["input"].tolist())

    df["captions"] = captions

    return df, name