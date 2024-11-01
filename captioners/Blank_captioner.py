import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math


import pandas as pd


def create_captions(df, batch_size):

    name = "BlankCaptioner"

    model_max_length = 512

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def get_captions(SMIs):
        all_caps = []

        for smis in tqdm(
            divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs) / batch_size)
        ):
            caps = [""] * len(smis)

            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["SMILES"].tolist())

    df["captions"] = captions

    return df, name
