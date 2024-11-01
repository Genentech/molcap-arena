import scipy.spatial

from GNEpropV2.data.data import MolecularDataset, index_predetermined_split
import pickle

import pandas as pd

import os

from collections import defaultdict

from tqdm import tqdm

all_smiles = set()


from GNEpropV2.data.data import MolecularDataset, index_predetermined_split
import pickle


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

    def load_splits(self):
        with open(os.path.join("splits/", self.name + "_splits.pkl"), "rb") as handle:
            b = pickle.load(handle)
            return b


datasets = [
    BenchmarkDataset(
        name="BBBP_clean", tasks=("p_np",), task_type="binary_classification"
    ),
    BenchmarkDataset(
        name="bace_clean", tasks=("Class",), task_type="binary_classification"
    ),
    BenchmarkDataset(
        name="clintox_clean", tasks=("CT_TOX",), task_type="binary_classification"
    ),
    BenchmarkDataset(
        name="esol_clean",
        tasks=("measured log solubility in mols per litre",),
        task_type="regression",
    ),
    BenchmarkDataset(name="freesolv_clean", tasks=("y",), task_type="regression"),
    BenchmarkDataset(name="lipo_clean", tasks=("exp",), task_type="regression"),
]

tasks = defaultdict(list)


fix_task = {
    "BBBP_clean": "BBBP",
    "clintox_clean": "ClinTox",
    "bace_clean": "BACE",
    "esol_clean": "ESOL",
    "freesolv_clean": "FreeSolv",
    "lipo_clean": "Lipo",
}

for i in datasets:
    dataset = i.load_dataset()

    all_smiles.update(dataset.smiles)

    for smi in dataset.smiles:
        tasks[smi].append(fix_task[i.name])

all_smiles = list(all_smiles)
tasks = [":".join(tasks[smi]) for smi in all_smiles]

df = pd.DataFrame({"SMILES": all_smiles, "tasks": tasks})

df.to_csv(f"all_smiles.csv", index=False)
