from sklearn import datasets, linear_model

from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import silhouette_score

import pickle
import numpy as np
from collections import defaultdict
import os, sys
import os.path as osp
from tqdm import tqdm
from math import comb
import random

import glob

from itertools import combinations
import argparse

from MolCapArena.rating.create_ELO_from_BT import get_battles

parser = argparse.ArgumentParser(description="Run an Experiment")
parser.add_argument("--dataset", type=str, default="BBBP")
parser.add_argument("--fold_idx", type=int, default=0)
parser.add_argument("--method", type=str, default="SVM")
args = parser.parse_args()


datasets = [args.dataset]
splits = [args.fold_idx]

method = args.method


dataset_to_task = {
    "BBBP": "BC",
    "BACE": "BC",
    "ClinTox": "BC",
    "ESOL": "R",
    "FreeSolv": "R",
    "Lipo": "R",
}


task = dataset_to_task[args.dataset]


cap_files = glob.glob(f"embeddings/{args.dataset}_*_0_train.pkl")
caption_sources = [
    cs.replace("embeddings/", "")
    .replace(args.dataset + "_", "", 1)
    .replace("_0_train.pkl", "")
    for cs in cap_files
]
caption_sources.remove("GNN")

train_embs = {}
valid_embs = {}
test_embs = {}
pref_embs = {}

for da in datasets:
    for cs in caption_sources + ["GNN"]:
        for i in splits:
            with open(f"embeddings/{da}_{cs}_{i}_train.pkl", "rb") as f:
                train_embs[f"{da}_{cs}_{i}"] = pickle.load(f)
            with open(f"embeddings/{da}_{cs}_{i}_val.pkl", "rb") as f:
                valid_embs[f"{da}_{cs}_{i}"] = pickle.load(f)
            with open(f"embeddings/{da}_{cs}_{i}_test.pkl", "rb") as f:
                test_embs[f"{da}_{cs}_{i}"] = pickle.load(f)
            with open(f"embeddings/{da}_{cs}_{i}_pref.pkl", "rb") as f:
                pref_embs[f"{da}_{cs}_{i}"] = pickle.load(f)


for da in datasets:
    for i in splits:

        smiles_tr = train_embs[f"{da}_GNN_{i}"][0]
        smiles_pref = pref_embs[f"{da}_GNN_{i}"][0]
        smiles_val = valid_embs[f"{da}_GNN_{i}"][0]
        smiles_te = test_embs[f"{da}_GNN_{i}"][0]

        combos = list(combinations(caption_sources, 2))
        random.shuffle(combos)

        for cs1, cs2 in tqdm(combos):

            cs1, cs2 = sorted([cs1, cs2])

            if os.path.exists(f"battles/{method}/{da}/{i}/{cs1}__{cs2}.csv"):
                continue

            smiles_tr1 = train_embs[f"{da}_{cs1}_{i}"][0]
            smiles_tr2 = train_embs[f"{da}_{cs2}_{i}"][0]
            order_tr1 = np.array([smiles_tr1.index(smi) for smi in smiles_tr])
            order_tr2 = np.array([smiles_tr2.index(smi) for smi in smiles_tr])
            X_tr1 = np.concatenate(
                [
                    train_embs[f"{da}_{cs1}_{i}"][1][order_tr1, :],
                    train_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            X_tr2 = np.concatenate(
                [
                    train_embs[f"{da}_{cs2}_{i}"][1][order_tr2, :],
                    train_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            X_tr = np.concatenate([X_tr1, X_tr2])
            Y_tr = np.concatenate(
                [
                    np.array(train_embs[f"{da}_{cs1}_{i}"][2])[order_tr1],
                    np.array(train_embs[f"{da}_{cs2}_{i}"][2])[order_tr2],
                ]
            ).squeeze()

            smiles_pref1 = pref_embs[f"{da}_{cs1}_{i}"][0]
            smiles_pref2 = pref_embs[f"{da}_{cs2}_{i}"][0]
            order_pref1 = np.array([smiles_pref1.index(smi) for smi in smiles_pref])
            order_pref2 = np.array([smiles_pref2.index(smi) for smi in smiles_pref])
            X_pref1 = np.concatenate(
                [
                    pref_embs[f"{da}_{cs1}_{i}"][1][order_pref1, :],
                    pref_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            X_pref2 = np.concatenate(
                [
                    pref_embs[f"{da}_{cs2}_{i}"][1][order_pref2, :],
                    pref_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            X_pref = np.concatenate([X_pref1, X_pref2])
            Y_pref = np.concatenate(
                [
                    np.array(pref_embs[f"{da}_{cs1}_{i}"][2])[order_pref1],
                    np.array(pref_embs[f"{da}_{cs2}_{i}"][2])[order_pref2],
                ]
            ).squeeze()

            smiles_val1 = valid_embs[f"{da}_{cs1}_{i}"][0]
            smiles_val2 = valid_embs[f"{da}_{cs2}_{i}"][0]
            order_val1 = np.array([smiles_val1.index(smi) for smi in smiles_val])
            order_val2 = np.array([smiles_val2.index(smi) for smi in smiles_val])
            X_val1 = np.concatenate(
                [
                    valid_embs[f"{da}_{cs1}_{i}"][1][order_val1, :],
                    valid_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            X_val2 = np.concatenate(
                [
                    valid_embs[f"{da}_{cs2}_{i}"][1][order_val2, :],
                    valid_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            X_val = np.concatenate([X_val1, X_val2])
            Y_val = np.concatenate(
                [
                    np.array(valid_embs[f"{da}_{cs1}_{i}"][2])[order_val1],
                    np.array(valid_embs[f"{da}_{cs2}_{i}"][2])[order_val2],
                ]
            ).squeeze()

            smiles_test1 = test_embs[f"{da}_{cs1}_{i}"][0]
            smiles_test2 = test_embs[f"{da}_{cs2}_{i}"][0]
            order_test1 = np.array([smiles_test1.index(smi) for smi in smiles_te])
            order_test2 = np.array([smiles_test2.index(smi) for smi in smiles_te])

            X_test1 = np.concatenate(
                [
                    test_embs[f"{da}_{cs1}_{i}"][1][order_test1, :],
                    test_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            X_test2 = np.concatenate(
                [
                    test_embs[f"{da}_{cs2}_{i}"][1][order_test2, :],
                    test_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )

            Y_test1 = (
                test_embs[f"{da}_{cs1}_{i}"][2]
                .cpu()
                .numpy()
                .squeeze()[order_test1]
                .squeeze()
            )
            Y_test2 = (
                test_embs[f"{da}_{cs2}_{i}"][2]
                .cpu()
                .numpy()
                .squeeze()[order_test2]
                .squeeze()
            )

            if method == "SVM":
                if task == "BC":
                    clf = make_pipeline(SVC(gamma="auto", probability=True))
                    clf.fit(X_pref, Y_pref)
                    Yp_pred1 = clf.predict_proba(X_test1)[:, 1]
                    Yp_pred2 = clf.predict_proba(X_test2)[:, 1]
                if task == "R":
                    clf = make_pipeline(SVR(gamma="auto"))
                    clf.fit(X_pref, Y_pref)
                    Yp_pred1 = clf.predict(X_test1)
                    Yp_pred2 = clf.predict(X_test2)

            elif method == "MLP":
                if task == "BC":
                    clf = make_pipeline(
                        MLPClassifier(hidden_layer_sizes=(100, 100, 100))
                    )
                    clf.fit(X_pref, Y_pref)
                    Yp_pred1 = clf.predict_proba(X_test1)[:, 1]
                    Yp_pred2 = clf.predict_proba(X_test2)[:, 1]
            error_dict = defaultdict(list)

            for smi, pred, act in zip(smiles_te, Yp_pred1, Y_test1):
                error_dict[smi].append((np.abs(act - pred), "", cs1))

            for smi, pred, act in zip(smiles_te, Yp_pred2, Y_test2):
                error_dict[smi].append((np.abs(act - pred), "", cs2))

            battles = get_battles(error_dict, [cs1, cs2])

            def print_winrate(df, models):

                num_models = len(models)
                model_to_id = {m: i for i, m in enumerate(models)}

                confusion_matrix = np.zeros((num_models))

                for _, row in df.iterrows():
                    e1, e2 = row["error_a"], row["error_b"]
                    m1, m2 = row["model_a"], row["model_b"]
                    if e1 < e2:
                        confusion_matrix[model_to_id[m1]] += 1
                    elif e2 < e1:
                        confusion_matrix[model_to_id[m2]] += 1

                winrate_matrix = confusion_matrix / (confusion_matrix.sum()) * 100

                print()
                print(models)
                print(winrate_matrix[0])
                print(
                    (
                        df[df["model_a"] == cs1]["error_a"].sum()
                        + df[df["model_b"] == cs1]["error_b"].sum()
                    )
                    / len(df),
                    (
                        df[df["model_a"] == cs2]["error_a"].sum()
                        + df[df["model_b"] == cs2]["error_b"].sum()
                    )
                    / len(df),
                )

            # print(cs1, cs2)
            # print_winrate(battles, [cs1, cs2])

            os.makedirs(f"battles/{method}/{da}/{i}/", exist_ok=True)
            battles.to_csv(f"battles/{method}/{da}/{i}/{cs1}__{cs2}.csv")
