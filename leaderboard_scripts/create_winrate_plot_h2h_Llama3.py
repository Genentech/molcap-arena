from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

from collections import defaultdict

import numpy as np

from MolCapArena.rating.get_ELO_head2head import get_battles_h2h


if True:
    datasets = ["ClinTox", "BBBP", "FreeSolv", "BACE", "Lipo", "ESOL"]
    splits = [0, 1, 2, 3, 4]
    data_folder = "battles"

if False:
    datasets = ["BBBP"]
    splits = [0, 1, 2, 3, 4]
    data_folder = "battles"


if "DATASETS" in os.environ:
    datasets = os.environ["DATASETS"].split()


battles = get_battles_h2h(data_folder, "SVM", datasets, splits)

battles = battles.dropna(subset=["model_a", "model_b"])

all_winrate_matrix = []

for dataset in datasets:

    df = battles[battles["Dataset"] == dataset]

    models = list(set(df["model_a"].unique()).union(set(df["model_b"].unique())))
    models = [m for m in models if "Llama3" in m]
    models = sorted(models)

    model_to_id = {m: i for i, m in enumerate(models)}

    num_models = len(models)

    confusion_matrix = np.zeros((num_models, num_models))

    for _, row in df.iterrows():
        e1, e2 = row["error_a"], row["error_b"]
        m1, m2 = row["model_a"], row["model_b"]
        if m1 not in models or m2 not in models:
            continue

        if e1 < e2:
            confusion_matrix[model_to_id[m1], model_to_id[m2]] += 1
        elif e2 < e1:
            confusion_matrix[model_to_id[m2], model_to_id[m1]] += 1

    confusion_matrix = confusion_matrix / 2
    cmt = confusion_matrix.transpose()
    winrate_matrix = confusion_matrix / (confusion_matrix + cmt) * 100
    np.fill_diagonal(winrate_matrix, np.nan)

    all_winrate_matrix.append(winrate_matrix)

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
    plt.savefig(f"plots/h2h/Llama3/{dataset}_winrate.png")

print(all_winrate_matrix)

if len(datasets) > 1:
    winrate_matrix = np.mean(np.stack(all_winrate_matrix), axis=0)

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
    plt.savefig(f"plots/h2h/Llama3/overall_winrate.png")
