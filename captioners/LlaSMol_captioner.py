import sys

sys.path.append("./captioners")
sys.path.append("./captioners/LLM4Chem")

import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import time


from LLM4Chem.generation import LlaSMolGeneration


def create_captions(df, batch_size):

    model_name = "osunlp/LlaSMol-Mistral-7B"

    name = model_name.split("/")[-1]

    model_max_length = 512

    def build_input(smiles_input):
        task_definition = (
            "Query: Describe this molecule: <SMILES> {} </SMILES> \n\n".format(
                smiles_input
            )
        )
        task_input = "Response: "

        model_input = task_definition + task_input
        return model_input

    df["input"] = df["SMILES"].progress_map(build_input)

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    generator = LlaSMolGeneration(model_name, device="cuda")

    def get_captions(inputs):
        all_caps = []

        for inps in tqdm(
            divide_chunks(inputs, batch_size), total=math.ceil(len(inputs) / batch_size)
        ):

            caps = generator.generate(inps, batch_size=len(inps))
            caps = [
                c["output"][0] if (c["output"] is not None) else "" for c in caps
            ]  # replace None with ''
            caps = [c.replace("<unk>", "").replace("</s>", "").strip() for c in caps]

            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["input"])

    df["captions"] = captions

    return df, name
