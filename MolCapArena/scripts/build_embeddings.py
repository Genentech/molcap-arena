import argparse
import os
import os.path as osp
import lightning as L
import sys
import torch
from lightning import Trainer, seed_everything

from MolCapArena.data.dataloaders import GraphDataModule, MultiCapDataModule
from MolCapArena.model.models import (
    GNNOnlyModel,
    CaptionOnlyModel,
    GNNCaptionModel,
    MoleculeTextModel,
)

import pickle

import subprocess

import pandas as pd

import time

from tqdm import tqdm

if __name__ == "__main__":
    torch.cuda.empty_cache()

    subprocess.run(["nvidia-smi"])

    config = {
        "pretrained_text_model": "michiyasunaga/BioLinkBERT-base",
        "trunc_length": 512,
        "num_warmup_steps": 1000,
        "max_epochs": 2,
        "batch_size": 128,
        "val_batch_size": None,
        "node_dim": 133,
        "edge_dim": 12,
        "hidden_dim_graph": 512,
        "num_mp_layers": 5,
        "num_readout_layers": 1,
        "dropout": 0.13,
        "aggr": "mean",
        "jk": "cat",
        "latent_size": 256,
        "validate_every_n": 1000,
        "lr": 2e-5,
        "data_module": "S1B",
        "ckpt_path": "ckpts/",
        "loss": "CLIP",
        "model": "GNN",
        "load_ckpt": None,
        "seed": 42,
    }

    parser = argparse.ArgumentParser(description="Biencoder")
    parser.add_argument(
        "--pretrained_text_model",
        default="michiyasunaga/BioLinkBERT-base",
        type=str,
        help="Which text encoder to use from HuggingFace",
    )
    parser.add_argument("--trunc_length", default=512, type=int)

    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--val_batch_size", default=None, type=int)

    parser.add_argument("--node_dim", default=133, type=int)
    parser.add_argument("--edge_dim", default=12, type=int)
    parser.add_argument("--hidden_dim_graph", default=512, type=int)
    parser.add_argument("--num_mp_layers", default=5, type=int)
    parser.add_argument("--num_readout_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.13, type=float)
    parser.add_argument("--aggr", default="mean", type=str)
    parser.add_argument("--jk", default="cat", type=str)

    parser.add_argument("--latent_size", default=256, type=int)
    parser.add_argument("--validate_every_n", default=1000, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--ckpt_path", default="ckpts/MOA/", type=str)
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="GNN", type=str)
    parser.add_argument("--mol_encoder_model", default="GNNOnlyModel", type=str)
    parser.add_argument("--text_encoder_model", default="CaptionOnlyModel", type=str)
    parser.add_argument(
        "--freeze_text_encoder", type=bool, action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        default="../ckpts/S1B/iter_test/CaptionGNN/78wnulit/best_val_checkpoint.ckpt",
        type=str,
    )

    parser.add_argument(
        "--GNN_checkpoint",
        default="ckpts/S1B/iter_test/BCE/tulun4e5/best_val_checkpoint.ckpt",
        type=str,
    )
    parser.add_argument(
        "--BERT_checkpoint",
        default="ckpts/S1B/iter_test/BCE/tulun4e5/best_val_checkpoint.ckpt",
        type=str,
    )

    parser.add_argument("--resume_wandb_run", default=None, type=str)

    parser.add_argument("--task", default=None, type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--data_module", type=str)
    parser.add_argument("--caption_source", type=str)
    parser.add_argument("--fold_idx", type=int)
    parser.add_argument("--text_latent_size", default=1024, type=int)

    args = parser.parse_args()

    if args.val_batch_size == None:
        args.val_batch_size = args.batch_size

    config = vars(args)
    print(config)

    if config["val_batch_size"] == None:
        config["val_batch_size"] = config["batch_size"]

    seed_everything(config["seed"])

    cs = config["caption_source"]

    if config["data_module"] == "BBBP":
        task = "BBBP"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
        )
    if config["data_module"] == "CapBBBP":
        task = "BBBP"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
            caption_source=config["caption_source"],
        )

    if config["data_module"] == "BACE":
        task = "BACE"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
        )
    if config["data_module"] == "CapBACE":
        task = "BACE"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
            caption_source=config["caption_source"],
        )

    if config["data_module"] == "ClinTox":
        task = "ClinTox"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
        )
    if config["data_module"] == "CapClinTox":
        task = "ClinTox"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
            caption_source=config["caption_source"],
        )

    if config["data_module"] == "ESOL":
        task = "ESOL"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
        )
    if config["data_module"] == "CapESOL":
        task = "ESOL"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
            caption_source=config["caption_source"],
        )

    if config["data_module"] == "FreeSolv":
        task = "FreeSolv"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
        )
    if config["data_module"] == "CapFreeSolv":
        task = "FreeSolv"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
            caption_source=config["caption_source"],
        )

    if config["data_module"] == "Lipo":
        task = "Lipo"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
        )
    if config["data_module"] == "CapLipo":
        task = "Lipo"
        output_dim = 1
        dm = GraphDataModule(
            config,
            pretrained_text_model=config["pretrained_text_model"],
            batch_size=config["batch_size"],
            trunc_length=config["trunc_length"],
            task=task,
            fold_idx=config["fold_idx"],
            caption_source=config["caption_source"],
        )

    config["task"] = task

    model_type = MoleculeTextModel

    if cs != "GNN":
        if config["model"] == "GNN":
            encoder_type = GNNOnlyModel
        elif config["model"] == "Caption":
            encoder_type = CaptionOnlyModel
        elif config["model"] == "CaptionGNN":
            encoder_type = GNNCaptionModel

        encoder = encoder_type(config, output_dim=output_dim)

        model = model_type.load_from_checkpoint(
            config["resume_from_checkpoint"],
            encoder=encoder,
            task="binary_classification",
            final_relu=False,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    else:
        encoder = GNNOnlyModel(config, output_dim=output_dim)
        model = model_type.load_from_checkpoint(
            config["GNN_checkpoint"],
            config=config,
            encoder=encoder,
            task="binary_classification",
            final_relu=False,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

    trainer = Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices="auto",
    )

    model.pred_embs = True

    dm.setup("")

    for dataloader, split in zip(
        [
            dm.test_dataloader(),
            dm.val_dataloader(),
            dm.pref_dataloader(),
            dm.train_dataloader(),
        ],
        ["test", "val", "pref", "train"],
    ):

        out = trainer.predict(model, dataloaders=dataloader)
        embs = [o[0] for o in out]
        batches = [o[1] for o in out]

        embs = torch.cat(embs)
        smiles = [y for x in batches for y in x["SMILES"]]
        acts = torch.cat([b["input"]["activity"] for b in batches])

        with open(f"embeddings/{task}_{cs}_{args.fold_idx}_{split}.pkl", "wb") as f:
            pickle.dump((smiles, embs, acts), f)
