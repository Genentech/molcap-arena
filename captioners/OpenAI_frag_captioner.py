import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

from openai import AzureOpenAI

import json
import os
import time

SLEEP_TIME = 0.2


def create_captions(
    df,
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8000/v1",
    sys_prompt="You are a helpful expert in medicinal chemistry. You are sharing your knowledge of known chemistry with a colleague.",
    prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    dump_file=None,
    dump_n=10,
):

    batch_size = 1

    name = model_name.split("/")[-1]

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

    client = AzureOpenAI(
        api_key=openai_api_key,
        azure_endpoint=openai_api_base,
        api_version="2024-02-01",
    )

    if dump_file is not None and os.path.exists(dump_file):
        prev_completions = json.load(open(dump_file))
    elif dump_file is not None:
        prev_completions = {}
    else:
        prev_completions = None

    def get_captions(SMIs):
        all_caps = []

        for i, smis in enumerate(
            tqdm(
                divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs) / batch_size)
            )
        ):
            smi = smis[0]
            if prev_completions is not None:
                if smi in prev_completions:
                    all_caps.append(prev_completions[smi])
                    continue

            history = get_history(smis[0])

            chat_response = client.chat.completions.create(
                model=model_name, messages=history
            )
            caps = [chat_response.choices[0].message.content]

            prev_completions[smi] = caps[0]

            all_caps.extend(caps)

            if (i + 1) % dump_n == 0 and prev_completions is not None:
                json.dump(prev_completions, open(dump_file, "w"), indent=4)

            time.sleep(SLEEP_TIME)

        return all_caps

    captions = get_captions(df["BRICS"].tolist())

    if dump_file is not None:
        json.dump(prev_completions, open(dump_file, "w"), indent=4)

    df["captions"] = captions

    return df, name
