from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

from collections import defaultdict

import numpy as np

from MolCapArena.rating.get_ELO_head2head import get_battles_h2h


datasets = ["BBBP", "BACE", "ClinTox", "ESOL", "FreeSolv", "Lipo"]
splits = [0, 1, 2, 3, 4]
data_folder = "battles"


if "DATASETS" in os.environ:
    datasets = os.environ["DATASETS"].split()


battles = get_battles_h2h(data_folder, "SVM", datasets, splits)
battles = battles.dropna(subset=["model_a", "model_b"])

groups = {
    "Llama3-8B-Generic": "Llama3-8B",
    "Llama3-8B-Drug": "Llama3-8B",
    "Llama3-8B-Bio": "Llama3-8B",
    "Llama3-8B-Chem": "Llama3-8B",
    "Llama3-8B-Quant": "Llama3-8B",
    "Llama3-8B-Frags-Generic": "Llama3-8B-Frags",
    "Llama3-8B-Frags-Drug": "Llama3-8B-Frags",
    "Llama3-8B-Frags-Bio": "Llama3-8B-Frags",
    "Llama3-8B-Frags-Chem": "Llama3-8B-Frags",
    "Llama3-8B-Frags-Quant": "Llama3-8B-Frags",
    "Llama3.1-8B-Generic": "Llama3.1-8B",
    "Llama3.1-8B-Drug": "Llama3.1-8B",
    "Llama3.1-8B-Bio": "Llama3.1-8B",
    "Llama3.1-8B-Chem": "Llama3.1-8B",
    "Llama3.1-8B-Quant": "Llama3.1-8B",
    "Llama3.1-8B-Frags-Generic": "Llama3.1-8B-Frags",
    "Llama3.1-8B-Frags-Drug": "Llama3.1-8B-Frags",
    "Llama3.1-8B-Frags-Bio": "Llama3.1-8B-Frags",
    "Llama3.1-8B-Frags-Chem": "Llama3.1-8B-Frags",
    "Llama3.1-8B-Frags-Quant": "Llama3.1-8B-Frags",
    "Llama3-70B-Generic": "Llama3-70B",
    "Llama3-70B-Drug": "Llama3-70B",
    "Llama3-70B-Bio": "Llama3-70B",
    "Llama3-70B-Chem": "Llama3-70B",
    "Llama3-70B-Quant": "Llama3-70B",
    "Llama3-70B-Frags-Generic": "Llama3-70B-Frags",
    "Llama3-70B-Frags-Drug": "Llama3-70B-Frags",
    "Llama3-70B-Frags-Bio": "Llama3-70B-Frags",
    "Llama3-70B-Frags-Chem": "Llama3-70B-Frags",
    "Llama3-70B-Frags-Quant": "Llama3-70B-Frags",
    "Llama3.1-70B-Generic": "Llama3.1-70B",
    "Llama3.1-70B-Drug": "Llama3.1-70B",
    "Llama3.1-70B-Bio": "Llama3.1-70B",
    "Llama3.1-70B-Chem": "Llama3.1-70B",
    "Llama3.1-70B-Quant": "Llama3.1-70B",
    "Llama3.1-70B-Frags-Generic": "Llama3.1-70B-Frags",
    "Llama3.1-70B-Frags-Drug": "Llama3.1-70B-Frags",
    "Llama3.1-70B-Frags-Bio": "Llama3.1-70B-Frags",
    "Llama3.1-70B-Frags-Chem": "Llama3.1-70B-Frags",
    "Llama3.1-70B-Frags-Quant": "Llama3.1-70B-Frags",
    "Gemma2-9B-Generic": "Gemma2-9B",
    "Gemma2-9B-Drug": "Gemma2-9B",
    "Gemma2-9B-Bio": "Gemma2-9B",
    "Gemma2-9B-Chem": "Gemma2-9B",
    "Gemma2-9B-Quant": "Gemma2-9B",
    "Gemma2-9B-Frags-Generic": "Gemma2-9B-Frags",
    "Gemma2-9B-Frags-Drug": "Gemma2-9B-Frags",
    "Gemma2-9B-Frags-Bio": "Gemma2-9B-Frags",
    "Gemma2-9B-Frags-Chem": "Gemma2-9B-Frags",
    "Gemma2-9B-Frags-Quant": "Gemma2-9B-Frags",
    "Gemma2-27B-Generic": "Gemma2-27B",
    "Gemma2-27B-Drug": "Gemma2-27B",
    "Gemma2-27B-Bio": "Gemma2-27B",
    "Gemma2-27B-Chem": "Gemma2-27B",
    "Gemma2-27B-Quant": "Gemma2-27B",
    "Gemma2-27B-Frags-Generic": "Gemma2-27B-Frags",
    "Gemma2-27B-Frags-Drug": "Gemma2-27B-Frags",
    "Gemma2-27B-Frags-Bio": "Gemma2-27B-Frags",
    "Gemma2-27B-Frags-Chem": "Gemma2-27B-Frags",
    "Gemma2-27B-Frags-Quant": "Gemma2-27B-Frags",
    "MistralNeMo-12B-Generic": "MistralNeMo-12B",
    "MistralNeMo-12B-Drug": "MistralNeMo-12B",
    "MistralNeMo-12B-Bio": "MistralNeMo-12B",
    "MistralNeMo-12B-Chem": "MistralNeMo-12B",
    "MistralNeMo-12B-Quant": "MistralNeMo-12B",
    "MistralNeMo-12B-Frags-Generic": "MistralNeMo-12B-Frags",
    "MistralNeMo-12B-Frags-Drug": "MistralNeMo-12B-Frags",
    "MistralNeMo-12B-Frags-Bio": "MistralNeMo-12B-Frags",
    "MistralNeMo-12B-Frags-Chem": "MistralNeMo-12B-Frags",
    "MistralNeMo-12B-Frags-Quant": "MistralNeMo-12B-Frags",
    "": "",
    "": "",
}


def to_group(model):
    return groups[model] if model in groups else model


battles["model_a"] = battles["model_a"].map(to_group)
battles["model_b"] = battles["model_b"].map(to_group)

all_winrate_matrix = []

for dataset in datasets:

    df = battles[battles["Dataset"] == dataset]

    models = list(set(df["model_a"].unique()).union(set(df["model_b"].unique())))
    models = sorted(models)
    model_to_id = {m: i for i, m in enumerate(models)}

    num_models = len(models)

    confusion_matrix = np.zeros((num_models, num_models))

    for _, row in df.iterrows():
        e1, e2 = row["error_a"], row["error_b"]
        m1, m2 = row["model_a"], row["model_b"]
        if e1 < e2:
            confusion_matrix[model_to_id[m1], model_to_id[m2]] += 1
        elif e2 < e1:
            confusion_matrix[model_to_id[m2], model_to_id[m1]] += 1

    confusion_matrix = confusion_matrix / 2
    cmt = confusion_matrix.transpose()
    winrate_matrix = confusion_matrix / (confusion_matrix + cmt) * 100
    np.fill_diagonal(winrate_matrix, np.nan)

    models_abbrev = [m.split("/")[-1] for m in models]

    disp = ConfusionMatrixDisplay(
        confusion_matrix=winrate_matrix,
        display_labels=models_abbrev,
    )
    fig, ax = plt.subplots(figsize=(25, 25), dpi=300)
    disp.plot(xticks_rotation="vertical", ax=ax)
    plt.ylabel("Model 1")
    plt.xlabel("Model 2")
    plt.tight_layout()
    plt.savefig(f"plots/h2h/groups/{dataset}_winrate.png")


df = battles

models = list(set(df["model_a"].unique()).union(set(df["model_b"].unique())))
models = sorted(models)
model_to_id = {m: i for i, m in enumerate(models)}

num_models = len(models)

confusion_matrix = np.zeros((num_models, num_models))

for _, row in df.iterrows():
    e1, e2 = row["error_a"], row["error_b"]
    m1, m2 = row["model_a"], row["model_b"]
    if e1 < e2:
        confusion_matrix[model_to_id[m1], model_to_id[m2]] += 1
    elif e2 < e1:
        confusion_matrix[model_to_id[m2], model_to_id[m1]] += 1

confusion_matrix = confusion_matrix / 2
cmt = confusion_matrix.transpose()
winrate_matrix = confusion_matrix / (confusion_matrix + cmt) * 100
np.fill_diagonal(winrate_matrix, np.nan)

models_abbrev = [m.split("/")[-1] for m in models]


disp = ConfusionMatrixDisplay(
    confusion_matrix=winrate_matrix,
    display_labels=models_abbrev,
)
fig, ax = plt.subplots(figsize=(25, 25), dpi=300)
disp.plot(xticks_rotation="vertical", ax=ax)
plt.ylabel("Model 1")
plt.xlabel("Model 2")
plt.tight_layout()
plt.savefig(f"plots/h2h/groups/overall_winrate.png")
