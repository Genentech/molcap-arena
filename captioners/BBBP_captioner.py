import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math


import pandas as pd

from rdkit import Chem


def canonicalize(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


def create_captions(df, batch_size):

    name = "BlankCaptioner"

    model_max_length = 512

    gt_df = pd.read_csv(
        "/gstore/data/resbioai/abxtext/evaluation_datasets/BBBP_clean.csv"
    )
    gt_df["smiles"] = gt_df["smiles"].apply(canonicalize)
    gt_df.set_index("smiles", inplace=True)

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def build_caption(smi):
        if smi in gt_df.index:
            if gt_df.loc[smi]["p_np"]:
                return "True"
            else:
                return "False"
        else:
            return "Unknown"

    def get_captions(SMIs):
        all_caps = []

        for smis in tqdm(
            divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs) / batch_size)
        ):
            caps = [build_caption(smi) for smi in smis]

            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["SMILES"].tolist())

    df["captions"] = captions

    return df, name
