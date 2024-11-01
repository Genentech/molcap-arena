from MolCapArena.rating.create_ELO_from_BT import get_ratings

from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm

import random

import numpy as np

import re


def get_battles_h2h(folder, method, datasets, folds):

    dfs = []

    num_files = []
    for da in datasets:
        for i in folds:

            path = f"{folder}/{method}/{da}/{i}/"

            files = [f for f in listdir(path) if isfile(join(path, f))]  # [:1000]
            files = [f for f in files if "__" in f]
            files = [f for f in files if "BBBP_captioner" not in f]
            num_files.append(len(files))
    assert (
        len(set(num_files)) == 1
    ), "Different number of battle files between datasets/folds"

    for da in datasets:
        for i in folds:

            path = f"{folder}/{method}/{da}/{i}/"

            files = [f for f in listdir(path) if isfile(join(path, f))]  # [:1000]
            files = [f for f in files if "__" in f]
            files = [f for f in files if "BBBP_captioner" not in f]

            bar = tqdm(files)
            bar.set_description(da + "_" + str(i))
            for file in bar:

                df = pd.read_csv(path + file)
                df["Dataset"] = da
                df["Fold"] = i

                # for BT stability, we sometimes need to random which side is the winner if 100% winrate
                # get 50% to swap
                if False:
                    mask = np.random.choice(a=[False, True], size=len(df))

                    # swap them
                    df.loc[mask, ["model_a", "model_b"]] = df.loc[
                        mask, ["model_b", "model_a"]
                    ].values
                    df.loc[mask, ["error_a", "error_b"]] = df.loc[
                        mask, ["error_b", "error_a"]
                    ].values
                    mask2 = df["winner"] == "model_a"
                    df.loc[mask & mask2, ["winner"]] = "model_b"
                    df.loc[mask & ~mask2, ["winner"]] = "model_a"

                dfs.append(df)

    battles = pd.concat(dfs)

    return battles


if __name__ == "__main__":

    battles = get_battles_h2h("battles", "SVM", datasets=["BBBP"], folds=[0])

    print(battles)

    bars = get_ratings(battles)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "expand_frame_repr",
        False,
    ):
        print(bars)
