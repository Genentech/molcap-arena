import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import torch

import os.path as osp
import sys

path = osp.dirname(osp.abspath(""))
# sys.path.append(path)
# sys.path.append(osp.join(path, "open_biomed"))

# load biomedgpt model
import json
import torch
from open_biomed.utils import fix_path_in_config
from open_biomed.models.multimodal import BioMedGPTV

from open_biomed.utils.chat_utils import Conversation


def create_captions(
    df,
):

    model_name = ("PharMolix/BioMedGPT-LM-7B",)

    batch_size = 1

    name = "BioMedGPT"

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    config = json.load(
        open("captioners/OpenBioMed/configs/encoders/multimodal/biomedgptv.json", "r")
    )
    fix_path_in_config(config, path)
    print("Config: ", config)

    device = torch.device("cuda:0")
    config["network"]["device"] = device
    model = BioMedGPTV(config["network"])

    return "Can't download from https://pan.baidu.com/s/1iAMBkuoZnNAylhopP5OgEg?pwd=7a6b#list/path=%2F "
    ckpt = torch.load("captioners/OpenBioMed/ckpts/fusion_ckpts/biomedgpt_10b.pth")
    model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    print("Finish loading model")

    prompt_sys = (
        "You are working as an excellent assistant in chemistry and molecule discovery. "
        + "Below a human expert gives the representation of a molecule or a protein. Answer questions about it. "
    )
    chat = Conversation(
        model=model,
        processor_config=config["data"],
        device=device,
        system=prompt_sys,
        roles=("Human", "Assistant"),
        sep="###",
        max_length=2048,
    )

    def get_captions(SMIs):
        all_caps = []

        for smis in tqdm(
            divide_chunks(SMIs, batch_size), total=math.ceil(len(SMIs) / batch_size)
        ):

            chat.reset()
            chat.append_molecule(smis[0])
            question = "Please describe this molecule."

            chat.ask(question)

            caps = [chat.answer()[0]]
            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["SMILES"].tolist())

    df["captions"] = captions

    return df, name
