from sklearn import datasets, linear_model

from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor


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

import pandas as pd

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

cap_files = glob.glob(f"embeddings/{args.dataset}_*_0_train.pkl")
caption_sources = [
    cs.replace("embeddings/", "")
    .replace(args.dataset + "_", "", 1)
    .replace("_0_train.pkl", "")
    for cs in cap_files
]


dataset_to_task = {
    "BBBP": "BC",
    "BACE": "BC",
    "ClinTox": "BC",
    "ESOL": "R",
    "FreeSolv": "R",
    "Lipo": "R",
}

task = dataset_to_task[args.dataset]


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

random.shuffle(caption_sources)

for da in datasets:
    for i in splits:

        smiles_tr = train_embs[f"{da}_GNN_{i}"][0]
        smiles_val = valid_embs[f"{da}_GNN_{i}"][0]
        smiles_te = test_embs[f"{da}_GNN_{i}"][0]
        smiles_pref = pref_embs[f"{da}_GNN_{i}"][0]

        pbar = tqdm(caption_sources + ["GNN"])
        for cs1 in pbar:

            pbar.set_description(cs1)

            if os.path.exists(f"battles/{method}/{da}/{i}/{cs1}.csv"):
                continue

            smiles_tr1 = train_embs[f"{da}_{cs1}_{i}"][0]
            order_tr1 = np.array([smiles_tr1.index(smi) for smi in smiles_tr])
            X_tr = np.concatenate(
                [
                    train_embs[f"{da}_{cs1}_{i}"][1][order_tr1, :],
                    train_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            Y_tr = np.array(train_embs[f"{da}_{cs1}_{i}"][2])[order_tr1].squeeze()

            smiles_pref1 = pref_embs[f"{da}_{cs1}_{i}"][0]
            order_pref1 = np.array([smiles_pref1.index(smi) for smi in smiles_pref])
            X_pref = np.concatenate(
                [
                    pref_embs[f"{da}_{cs1}_{i}"][1][order_pref1, :],
                    pref_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            Y_pref = np.array(pref_embs[f"{da}_{cs1}_{i}"][2])[order_pref1].squeeze()

            smiles_val1 = valid_embs[f"{da}_{cs1}_{i}"][0]
            order_val1 = np.array([smiles_val1.index(smi) for smi in smiles_val])
            X_val = np.concatenate(
                [
                    valid_embs[f"{da}_{cs1}_{i}"][1][order_val1, :],
                    valid_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            Y_val = np.array(valid_embs[f"{da}_{cs1}_{i}"][2])[order_val1].squeeze()

            smiles_test1 = test_embs[f"{da}_{cs1}_{i}"][0]
            order_test1 = np.array([smiles_test1.index(smi) for smi in smiles_te])
            X_test = np.concatenate(
                [
                    test_embs[f"{da}_{cs1}_{i}"][1][order_test1, :],
                    test_embs[f"{da}_GNN_{i}"][1],
                ],
                axis=1,
            )
            Y_test = (
                test_embs[f"{da}_{cs1}_{i}"][2]
                .cpu()
                .numpy()
                .squeeze()[order_test1]
                .squeeze()
            )

            if cs1 == "GNN":
                X_tr = X_tr[:, X_tr.shape[1] // 2 :]
                X_pref = X_pref[:, X_pref.shape[1] // 2 :]
                X_val = X_val[:, X_val.shape[1] // 2 :]
                X_test = X_test[:, X_test.shape[1] // 2 :]

            if "train1" in method:
                X_h, Y_h = X_tr, Y_tr
            else:
                X_h, Y_h = X_pref, Y_pref

            if method.startswith("SVM"):
                if task == "BC":
                    clf = make_pipeline(SVC(gamma="auto", probability=True))
                    clf.fit(X_pref, Y_pref)
                    Yte_pred = clf.predict_proba(X_test)[:, 1]
                if task == "R":
                    clf = make_pipeline(SVR(gamma="auto"))
                    clf.fit(X_pref, Y_pref)
                    Yte_pred = clf.predict(X_test)

            error_dict = defaultdict(list)

            res = pd.DataFrame(
                {
                    "Source Model": [cs1] * len(Y_test),
                    "Prediction": Yte_pred,
                    "Activity": Y_test,
                }
            )

            os.makedirs(f"battles/{method}/{da}/{i}/", exist_ok=True)
            res.to_csv(
                f"battles/{method}/{da}/{i}/{cs1}.csv",
                index=False,
                float_format="%.10f",
            )  # scientific notation is causing issues so don't use
