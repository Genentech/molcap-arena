import lightning as L
from lightning import LightningDataModule
import sys
import torch
import torch.utils.data
from MolCapArena.data.processing import smiles_to_data, smiles_list_to_mols
from collections.abc import Mapping
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from transformers import AutoTokenizer
from typing import Any, List, Optional, Sequence, Union
from tqdm import tqdm

import pandas as pd

from sklearn.model_selection import train_test_split

from rdkit.Chem.inchi import MolToInchi
from rdkit import Chem

import numpy as np

from collections import defaultdict
from sklearn.utils import resample

import random

import os
import os.path as osp

import pickle

from MolCapArena.data.processing import (
    MolecularDataset,
    index_predetermined_split,
    MolecularSubset,
)


def canonicalize(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


class CustomCollater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, Batch):

            rv = [b.to_data_list() for b in batch]
            rv = [item for sublist in rv for item in sublist]
            return Batch.from_data_list(rv)
        elif isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        # elif isinstance(elem, TensorFrame):
        #    return torch_frame.cat(batch, along='row')
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        # if isinstance(self.dataset, OnDiskDataset):
        #    return self(self.dataset.multi_get(batch))
        return self(batch)


class CustomDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = CustomCollater(dataset, follow_batch, exclude_keys)

        # if isinstance(dataset, OnDiskDataset):
        #    dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )


class BenchmarkDataset:
    def __init__(self, name, tasks, task_type):
        self.name = name
        self.tasks = tasks
        self.task_type = task_type

    def load_dataset(self):
        dataset = MolecularDataset.load_csv_dataset(
            os.path.join("splits/", self.name + ".csv"),
            smiles_column_name="smiles",
            y_column_names=self.tasks,
        )
        return dataset

    def load_splits(self, four_version=False):
        if not four_version:
            dataset_path = os.path.join("splits/", self.name + "_splits.pkl")
        else:
            dataset_path = os.path.join("splits/", self.name + "_4" + "_splits.pkl")

        with open(dataset_path, "rb") as handle:
            b = pickle.load(handle)
            return b


def index_predetermined_split_n(dataset: MolecularDataset, seed: int = 0):
    split_indices = dataset.index_predetermined[seed]
    list_ix = split_indices

    return [MolecularSubset(dataset, i_ix) for i_ix in list_ix]


class GraphDataset(Dataset):

    def __init__(
        self, data, tokenizer, trunc_length=512, split=None, caption_data=None
    ):
        self.data = data
        self.tokenizer = tokenizer

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length
        self.caption_data = caption_data

    def len(self):
        return len(self.data)

    def get(self, idx):
        SMILES = self.data.smiles[idx]
        y = self.data.y[idx]

        act = torch.tensor(y, dtype=float).float()

        molecule_input = smiles_to_data(SMILES)

        if self.caption_data is not None:
            text_caption = self.caption_data.loc[SMILES]["captions"]
            if type(text_caption) != str:  # it's a blank
                text_caption = ""
            caption = self.tokenizer(
                text_caption,
                truncation=True,
                max_length=self.trunc_length,
                padding="max_length",
                return_tensors="pt",
            )
            for key in caption:
                caption[key] = caption[key].squeeze()
        else:
            text_caption = caption = "None"

        rv = {
            "SMILES": SMILES,
            "task_id": 0,
            "caption": text_caption,
            "input": {
                "molecule": molecule_input,
                "activity": act,
                "caption": caption,
            },
        }
        return rv


class GraphDataModule(LightningDataModule):

    def __init__(
        self,
        config,
        pretrained_text_model,
        batch_size=128,
        trunc_length=512,
        task="BBBP",
        fold_idx=0,
        caption_path="captions/",
        caption_source=None,
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.pretrained_text_model = pretrained_text_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.task = task
        self.caption_path = caption_path
        self.config = config
        self.fold_idx = fold_idx
        self.caption_source = caption_source

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_text_model)

        if self.task == "BBBP":
            i = BenchmarkDataset(
                name="BBBP_clean", tasks=("p_np",), task_type="binary_classification"
            )
        elif self.task == "BACE":
            i = BenchmarkDataset(
                name="bace_clean", tasks=("Class",), task_type="binary_classification"
            )
        elif self.task == "ClinTox":
            i = BenchmarkDataset(
                name="clintox_clean",
                tasks=("CT_TOX",),
                task_type="binary_classification",
            )
        elif self.task == "ESOL":
            i = BenchmarkDataset(
                name="esol_clean",
                tasks=("measured log solubility in mols per litre",),
                task_type="regression",
            )
        elif self.task == "FreeSolv":
            i = BenchmarkDataset(
                name="freesolv_clean", tasks=("y",), task_type="regression"
            )
        elif self.task == "Lipo":
            i = BenchmarkDataset(
                name="lipo_clean", tasks=("exp",), task_type="regression"
            )

        dataset = i.load_dataset()
        dataset_splits = i.load_splits(four_version=True)
        dataset.index_predetermined = dataset_splits

        train_set, pref_set, val_set, test_set = index_predetermined_split_n(
            dataset, seed=self.fold_idx
        )

        def contains_task(tasks_str):
            tasks_list = tasks_str.split(":")
            return self.task in tasks_list

        if self.caption_source != None:
            caption_data = pd.read_csv(self.caption_path + self.caption_source + ".csv")
            if "tasks" in caption_data.keys():
                caption_data = caption_data[caption_data["tasks"].apply(contains_task)]
            caption_data.set_index("SMILES", inplace=True)
            if (
                "MOA_caption" in caption_data.keys()
                and "captions" not in caption_data.keys()
            ):
                caption_data["captions"] = caption_data["MOA_caption"]
        else:
            caption_data = None

        self.train_ds = GraphDataset(
            train_set,
            self.tokenizer,
            split="train",
            trunc_length=self.trunc_length,
            caption_data=caption_data,
        )
        self.pref_ds = GraphDataset(
            pref_set,
            self.tokenizer,
            split="train",
            trunc_length=self.trunc_length,
            caption_data=caption_data,
        )
        self.valid_ds = GraphDataset(
            val_set,
            self.tokenizer,
            split="valid",
            trunc_length=self.trunc_length,
            caption_data=caption_data,
        )
        self.test_ds = GraphDataset(
            test_set,
            self.tokenizer,
            split="test",
            trunc_length=self.trunc_length,
            caption_data=caption_data,
        )

    def train_dataloader(self):
        return CustomDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def pref_dataloader(self):
        return CustomDataLoader(
            self.pref_ds, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return CustomDataLoader(
            self.valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def test_dataloader(self):
        return CustomDataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def teardown(self, stage: str):
        pass


class MultiCapDataset(Dataset):

    def __init__(self, data, tokenizer, trunc_length=512, split=None):
        self.data = data
        self.tokenizer = tokenizer

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length

    def len(self):
        return len(self.data)

    def get(self, idx):
        d = self.data.iloc[idx]
        SMILES = d["SMILES"]
        act = float(d["Activity"])
        caption = d["caption"]
        sm = d["Source Model"]

        molecule_input = smiles_to_data(SMILES)

        caption = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.trunc_length,
            padding="max_length",
            return_tensors="pt",
        )
        for key in caption:
            caption[key] = caption[key].squeeze()

        return {
            "SMILES": SMILES,
            "task_id": 0,
            "caption": d["caption"],
            "Source Model": sm,
            "input": {
                "molecule": molecule_input,
                "activity": torch.tensor(act).unsqueeze(0),
                "caption": caption,
            },
        }


class MultiCapDataModule(LightningDataModule):

    def __init__(
        self,
        pretrained_text_model,
        batch_size=128,
        trunc_length=512,
        NCE_ver=False,
        task="BBBP",
        caption_file="/home/edwarc24/data/antibioticbiencoder/captions/",
        fold_idx=0,
        add_blank=False,
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.pretrained_text_model = pretrained_text_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.task = task
        self.NCE_ver = NCE_ver
        self.caption_file = caption_file
        self.add_blank = add_blank
        self.fold_idx = fold_idx

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_text_model)

        # self.data = pd.read_csv(self.data_file)

        if self.task == "BBBP":
            i = BenchmarkDataset(
                name="BBBP_clean", tasks=("p_np",), task_type="binary_classification"
            )
        elif self.task == "BACE":
            i = BenchmarkDataset(
                name="bace_clean", tasks=("Class",), task_type="binary_classification"
            )
        elif self.task == "ClinTox":
            i = BenchmarkDataset(
                name="clintox_clean",
                tasks=("CT_TOX",),
                task_type="binary_classification",
            )
        elif self.task == "Tox21":
            i = BenchmarkDataset(
                name="tox21_clean",
                tasks=(
                    "NR-AR",
                    "NR-AR-LBD",
                    "NR-AhR",
                    "NR-Aromatase",
                    "NR-ER",
                    "NR-ER-LBD",
                    "NR-PPAR-gamma",
                    "SR-ARE",
                    "SR-ATAD5",
                    "SR-HSE",
                    "SR-MMP",
                    "SR-p53",
                ),
                task_type="binary_classification",
            )
        elif self.task == "Sider":
            BenchmarkDataset(
                name="sider_clean",
                tasks=(
                    "Hepatobiliary disorders",
                    "Metabolism and nutrition disorders",
                    "Product issues",
                    "Eye disorders",
                    "Investigations",
                    "Musculoskeletal and connective tissue disorders",
                    "Gastrointestinal disorders",
                    "Social circumstances",
                    "Immune system disorders",
                    "Reproductive system and breast disorders",
                    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                    "General disorders and administration site conditions",
                    "Endocrine disorders",
                    "Surgical and medical procedures",
                    "Vascular disorders",
                    "Blood and lymphatic system disorders",
                    "Skin and subcutaneous tissue disorders",
                    "Congenital, familial and genetic disorders",
                    "Infections and infestations",
                    "Respiratory, thoracic and mediastinal disorders",
                    "Psychiatric disorders",
                    "Renal and urinary disorders",
                    "Pregnancy, puerperium and perinatal conditions",
                    "Ear and labyrinth disorders",
                    "Cardiac disorders",
                    "Nervous system disorders",
                    "Injury, poisoning and procedural complications",
                ),
                task_type="binary_classification",
            )
        elif self.task == "OPV":
            i = BenchmarkDataset(
                name="opv_clean",
                tasks=(
                    "HOMO",
                    "LUMO",
                    "electrochemical_gap",
                    "optical_gap",
                    "PCE",
                    "V_OC",
                    "J_SC",
                    "fill_factor",
                ),
                task_type="regression",
            )
        elif self.task == "ESOL":
            i = BenchmarkDataset(
                name="esol_clean",
                tasks=("measured log solubility in mols per litre",),
                task_type="regression",
            )
        elif self.task == "FreeSolv":
            i = BenchmarkDataset(
                name="freesolv_clean", tasks=("y",), task_type="regression"
            )
        elif self.task == "Lipo":
            i = BenchmarkDataset(
                name="lipo_clean", tasks=("exp",), task_type="regression"
            )

        dataset = i.load_dataset()
        dataset_splits = i.load_splits(four_version=True)
        dataset.index_predetermined = dataset_splits

        train_set, pref_set, val_set, test_set = index_predetermined_split_n(
            dataset, seed=self.fold_idx
        )

        train_data_smi = train_set.smiles
        valid_data_smi = val_set.smiles
        test_data_smi = test_set.smiles

        label_map = {}
        for ds in [train_set, val_set, test_set]:
            for smi, y in zip(ds.smiles, ds.y):
                label_map[smi] = y

        caption_path = self.caption_file

        cap_files = [
            osp.join(caption_path, f)
            for f in os.listdir(caption_path)
            if osp.isfile(osp.join(caption_path, f))
            and osp.join(caption_path, f).endswith("csv")
        ]

        new_train_rows = []
        new_val_rows = []
        new_test_rows = []

        for cf in cap_files:
            df = pd.read_csv(cf)
            name = cf[:-4].split("/")[-1]

            for row in df.iterrows():
                if type(row[1]["captions"]) != str:  # it's a blank .
                    row[1]["captions"] = ""

                smi = row[1]["SMILES"]
                if smi in train_data_smi:
                    new_train_rows.append(
                        [smi, row[1]["captions"], label_map[smi], name]
                    )
                if smi in valid_data_smi:
                    new_val_rows.append([smi, row[1]["captions"], label_map[smi], name])
                if smi in test_data_smi:
                    new_test_rows.append(
                        [smi, row[1]["captions"], label_map[smi], name]
                    )

        if self.add_blank:
            name = "BlankCaption"
            for row in df.iterrows():

                smi = row[1]["SMILES"]
                if smi in train_data_smi:
                    new_train_rows.append([smi, "", label_map[smi], name])
                if smi in valid_data_smi:
                    new_val_rows.append([smi, "", label_map[smi], name])
                if smi in test_data_smi:
                    new_test_rows.append([smi, "", label_map[smi], name])

        cols = ["SMILES", "caption", "Activity", "Source Model"]
        train_data = pd.DataFrame(new_train_rows, columns=cols)
        valid_data = pd.DataFrame(new_val_rows, columns=cols)
        test_data = pd.DataFrame(new_test_rows, columns=cols)

        self.train_ds = MultiCapDataset(
            train_data, self.tokenizer, split="train", trunc_length=self.trunc_length
        )
        self.valid_ds = MultiCapDataset(
            valid_data, self.tokenizer, split="valid", trunc_length=self.trunc_length
        )
        self.test_ds = MultiCapDataset(
            test_data, self.tokenizer, split="test", trunc_length=self.trunc_length
        )

    def train_dataloader(self):
        return CustomDataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return CustomDataLoader(
            self.valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def test_dataloader(self):
        return CustomDataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def teardown(self, stage: str):
        pass


class CapOnlyDataset(Dataset):

    def __init__(self, data, tokenizer, trunc_length=512, split=None, cap_to_id=None):
        self.data = data
        self.tokenizer = tokenizer

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length
        self.cap_to_id = cap_to_id

    def len(self):
        return len(self.data)

    def get(self, idx):
        d = self.data.iloc[idx]
        SMILES = d["SMILES"]
        caption = d["caption"]
        sm = d["Source Model"]

        molecule_input = smiles_to_data(SMILES)

        caption = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.trunc_length,
            padding="max_length",
            return_tensors="pt",
        )
        for key in caption:
            caption[key] = caption[key].squeeze()

        return {
            "SMILES": SMILES,
            "task_id": 0,
            "caption": d["caption"],
            "Source Model": sm,
            "input": {
                "molecule": molecule_input,
                "caption": caption,
                "activity": self.cap_to_id[sm],
            },
        }


class MultiCapOnlyDataModule(LightningDataModule):

    def __init__(
        self,
        pretrained_text_model,
        batch_size=128,
        trunc_length=512,
        NCE_ver=False,
        tasks=None,
        caption_file="captions/",
        fold_idx=0,
        add_blank=False,
        caption_source=None,
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.pretrained_text_model = pretrained_text_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.tasks = tasks
        self.NCE_ver = NCE_ver
        self.caption_file = caption_file
        self.add_blank = add_blank
        self.fold_idx = fold_idx
        self.caption_source = caption_source

        caption_path = self.caption_file

        self.cap_files = [
            osp.join(caption_path, f)
            for f in os.listdir(caption_path)
            if osp.isfile(osp.join(caption_path, f))
            and osp.join(caption_path, f).endswith("csv")
        ]

        self.names = [cf[:-4].split("/")[-1] for cf in self.cap_files]
        self.num_caption_sources = len(self.names)
        self.cap_to_id = {cap: i for i, cap in enumerate(self.names)}
        for cap in other_names:
            self.cap_to_id[cap] = -1

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_text_model)

        new_train_rows = []
        new_val_rows = []
        new_test_rows = []

        pbar = tqdm(self.tasks)
        for task in pbar:
            pbar.set_description(task)
            if task == "BBBP":
                i = BenchmarkDataset(
                    name="BBBP_clean",
                    tasks=("p_np",),
                    task_type="binary_classification",
                )
            elif task == "BACE":
                i = BenchmarkDataset(
                    name="bace_clean",
                    tasks=("Class",),
                    task_type="binary_classification",
                )
            elif task == "ClinTox":
                i = BenchmarkDataset(
                    name="clintox_clean",
                    tasks=("CT_TOX",),
                    task_type="binary_classification",
                )
            elif task == "ESOL":
                i = BenchmarkDataset(
                    name="esol_clean",
                    tasks=("measured log solubility in mols per litre",),
                    task_type="regression",
                )
            elif task == "FreeSolv":
                i = BenchmarkDataset(
                    name="freesolv_clean", tasks=("y",), task_type="regression"
                )
            elif task == "Lipo":
                i = BenchmarkDataset(
                    name="lipo_clean", tasks=("exp",), task_type="regression"
                )

            dataset = i.load_dataset()
            dataset_splits = i.load_splits(four_version=True)
            dataset.index_predetermined = dataset_splits

            train_set, pref_set, val_set, test_set = index_predetermined_split_n(
                dataset, seed=self.fold_idx
            )

            train_data_smi = train_set.smiles
            valid_data_smi = val_set.smiles
            test_data_smi = test_set.smiles

            if self.caption_source != None:
                pbar2 = tqdm([self.caption_file + self.caption_source + ".csv"])
            else:
                pbar2 = tqdm(self.cap_files)
            for cf in pbar2:
                df = pd.read_csv(cf)
                name = cf[:-4].split("/")[-1]
                pbar2.set_description(name)

                for row in df.iterrows():
                    if type(row[1]["captions"]) != str:  # it's a blank .
                        row[1]["captions"] = ""

                    smi = row[1]["SMILES"]
                    if smi in train_data_smi:
                        new_train_rows.append([smi, row[1]["captions"], name])
                    if smi in valid_data_smi:
                        new_val_rows.append([smi, row[1]["captions"], name])
                    if smi in test_data_smi:
                        new_test_rows.append([smi, row[1]["captions"], name])

            if self.add_blank:
                name = "BlankCaption"
                for row in df.iterrows():

                    smi = row[1]["SMILES"]
                    if smi in train_data_smi:
                        new_train_rows.append([smi, "", label_map[smi], name])
                    if smi in valid_data_smi:
                        new_val_rows.append([smi, "", label_map[smi], name])
                    if smi in test_data_smi:
                        new_test_rows.append([smi, "", label_map[smi], name])

        cols = ["SMILES", "caption", "Source Model"]
        train_data = pd.DataFrame(new_train_rows, columns=cols)
        valid_data = pd.DataFrame(new_val_rows, columns=cols)
        test_data = pd.DataFrame(new_test_rows, columns=cols)

        self.train_ds = CapOnlyDataset(
            train_data,
            self.tokenizer,
            split="train",
            trunc_length=self.trunc_length,
            cap_to_id=self.cap_to_id,
        )
        self.valid_ds = CapOnlyDataset(
            valid_data,
            self.tokenizer,
            split="valid",
            trunc_length=self.trunc_length,
            cap_to_id=self.cap_to_id,
        )
        self.test_ds = CapOnlyDataset(
            test_data,
            self.tokenizer,
            split="test",
            trunc_length=self.trunc_length,
            cap_to_id=self.cap_to_id,
        )

    def train_dataloader(self):
        return CustomDataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return CustomDataLoader(
            self.valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def test_dataloader(self):
        return CustomDataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def teardown(self, stage: str):
        pass
