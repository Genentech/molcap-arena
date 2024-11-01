from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

import math

from collections import defaultdict

import numpy as np

import os

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

dataset_to_task = {
    "BBBP": "BC",
    "BACE": "BC",
    "ClinTox": "BC",
    "ESOL": "R",
    "FreeSolv": "R",
    "Lipo": "R",
}


if True:
    datasets = ["BBBP", "BACE", "ClinTox", "ESOL", "FreeSolv", "Lipo"]
    splits = [0, 1, 2, 3, 4]
    method = "SVM"
    data_folder = "battles"


if False:
    datasets = ["ClinTox"]
    splits = [0, 1, 2, 3, 4]
    method = "SVM"
    data_folder = "battles"


if "DATASETS" in os.environ:
    datasets = os.environ["DATASETS"].split()

import glob

cap_files = glob.glob(f"{data_folder}/{method}/{datasets[0]}/{splits[0]}/*.csv")
caption_sources = [cf.split("/")[-1].split(".csv")[0] for cf in cap_files]
caption_sources = list(
    set([cs for cs2 in caption_sources for cs in cs2.split("__")])
) + ["GNN"]
caption_sources.remove("BBBP_captioner")


AUCs = defaultdict(list)
APs = defaultdict(list)
Ls = defaultdict(list)
AEs = defaultdict(list)
PEARs = defaultdict(list)
SPEARs = defaultdict(list)
MSEs = defaultdict(list)
R2s = defaultdict(list)
MAEs = defaultdict(list)

source_models = set()


for dataset in datasets:
    task = dataset_to_task[dataset]

    for cs in caption_sources:
        if not all(
            [
                os.path.exists(f"{data_folder}/{method}/{dataset}/{i}/{cs}.csv")
                for i in splits
            ]
        ):
            continue

        for i in splits:

            df = pd.read_csv(f"{data_folder}/{method}/{dataset}/{i}/{cs}.csv")

            num_features = len(
                [col for col in df.columns if col.startswith("Activity")]
            )

            for j in range(num_features):

                if task == "BC":
                    df["Error"] = (df["Activity"] - df["Prediction"]) * df[
                        "Activity"
                    ] + (1 - df["Activity"]) * (df["Prediction"] - df["Activity"])
                if task == "MBC" or task == "MBC_nan":
                    df[f"Error_{j}"] = (
                        df[f"Activity_{j}"] - df[f"Prediction_{j}"]
                    ) * df[f"Activity_{j}"] + (1 - df[f"Activity_{j}"]) * (
                        df[f"Prediction_{j}"] - df[f"Activity_{j}"]
                    )
                elif task == "R":
                    df["Error"] = np.absolute((df["Activity"] - df["Prediction"]))
                elif task == "MR" or task == "MR_nan":
                    df[f"Error_{j}"] = np.absolute(
                        (df[f"Activity_{j}"] - df[f"Prediction_{j}"])
                    )

                error_dict = defaultdict(list)

                for row in df.iterrows():
                    d = row[1]
                    if num_features > 1:
                        error_dict[d["Source Model"]].append(
                            (
                                d[f"Prediction_{j}"],
                                d[f"Activity_{j}"],
                                "",
                                d["Source Model"],
                                d[f"Error_{j}"],
                            )
                        )
                    else:
                        error_dict[d["Source Model"]].append(
                            (
                                d["Prediction"],
                                d["Activity"],
                                "",
                                d["Source Model"],
                                d["Error"],
                            )
                        )

                    source_models.add(d["Source Model"])

                if task == "BC" or task == "MBC" or task == "MBC_nan":
                    for sm in sorted(error_dict.keys()):

                        pred = np.array([a[0] for a in error_dict[sm]])
                        gt = np.array([a[1] for a in error_dict[sm]])

                        mask = ~np.isnan(gt)
                        pred = pred[mask]
                        gt = gt[mask].astype(int)

                        rocauc = roc_auc_score(gt, pred)

                        AUCs[sm].append(rocauc)

                    for sm in sorted(error_dict.keys()):

                        pred = np.array([a[0] for a in error_dict[sm]])
                        gt = np.array([a[1] for a in error_dict[sm]])

                        mask = ~np.isnan(gt)
                        pred = pred[mask]
                        gt = gt[mask].astype(int)

                        ap = average_precision_score(gt, pred)

                        APs[sm].append(ap)

                    for sm in sorted(error_dict.keys()):

                        pred = np.array([a[0] for a in error_dict[sm]])
                        gt = np.array([a[1] for a in error_dict[sm]])

                        mask = ~np.isnan(gt)
                        pred = pred[mask]
                        gt = gt[mask].astype(int)

                        l = log_loss(gt, pred)

                        Ls[sm].append(l)

                elif task == "R" or task == "MR" or task == "MR_nan":
                    for sm in sorted(error_dict.keys()):

                        pred = np.array([a[0] for a in error_dict[sm]])
                        gt = np.array([a[1] for a in error_dict[sm]])

                        mask = ~np.isnan(gt)
                        pred = pred[mask]
                        gt = gt[mask]

                        pear = pearsonr(gt, pred).statistic
                        spear = spearmanr(gt, pred).statistic
                        mse = mean_squared_error(gt, pred)
                        r2 = r2_score(gt, pred)
                        mae = mean_absolute_error(gt, pred)

                        PEARs[sm].append(pear)
                        SPEARs[sm].append(spear)
                        R2s[sm].append(r2)
                        MSEs[sm].append(mse)
                        MAEs[sm].append(mae)

                for sm in sorted(error_dict.keys()):

                    e = np.array([a[4] for a in error_dict[sm]])

                    mask = ~np.isnan(e)
                    e = e[mask]

                    e = np.mean(e)

                    AEs[sm].append(e)


for sm in APs:
    APs[sm] = np.mean(APs[sm])
    AUCs[sm] = np.mean(AUCs[sm])
    Ls[sm] = np.mean(Ls[sm])

for sm in AEs:
    AEs[sm] = np.mean(AEs[sm])

for sm in PEARs:
    PEARs[sm] = np.mean(PEARs[sm])
    SPEARs[sm] = np.mean(SPEARs[sm])
    MSEs[sm] = np.mean(MSEs[sm])
    R2s[sm] = np.mean(R2s[sm])
    MAEs[sm] = np.mean(MAEs[sm])

models = sorted(list(set(caption_sources)))

pd.set_option("display.precision", 3)

if True:  # get correlation with ELO
    from MolCapArena.rating.create_ELO_from_BT import get_ratings
    from MolCapArena.rating.get_ELO_head2head import get_battles_h2h

    battles = get_battles_h2h(data_folder, "SVM", datasets, splits)
    bars = get_ratings(battles)

    bars = bars._append({"model": "GNN", "rating": np.nan}, ignore_index=True)

    bars["ROC-AUC"] = bars["model"].map(AUCs)
    bars["Loss"] = bars["model"].map(Ls)
    bars["Avg. Error"] = bars["model"].map(AEs)
    bars["Average Precision"] = bars["model"].map(APs)
    bars["ROC-AUC"] = bars["ROC-AUC"] * 100
    bars["Average Precision"] = bars["Average Precision"] * 100

    bars["Pearson R"] = bars["model"].map(PEARs)
    bars["Spearman R"] = bars["model"].map(SPEARs)
    bars["R^2"] = bars["model"].map(R2s)
    bars["MSE"] = bars["model"].map(MSEs)
    bars["MAE"] = bars["model"].map(MAEs)

    bars = bars.set_index("model")

    bars_na = bars.apply(pd.to_numeric, errors="coerce")
    bars_na = bars_na.dropna(axis=1, how="all")
    bars_na = bars_na.dropna()

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "expand_frame_repr",
        False,
    ):
        print("Pearson")
        print(bars_na.corr(numeric_only=True))
        print("Spearman")
        print(bars_na.corr(numeric_only=True, method="spearman"))

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "expand_frame_repr",
        False,
    ):
        print(bars)

    if False:
        ds = str(datasets)
        with open(f"latex_files/{ds}_Pearson.txt", "w") as text_file:
            tmp = bars_na.corr(numeric_only=True).to_latex(
                index=True, float_format="{:.3f}".format
            )
            text_file.write(tmp)
        bars_na.corr(numeric_only=True).to_csv(
            f"latex_files/{ds}_Pearson.csv", index=True
        )
        with open(f"latex_files/{ds}_Spearman.txt", "w") as text_file:
            tmp = bars_na.corr(numeric_only=True, method="spearman").to_latex(
                index=True, float_format="{:.3f}".format
            )
            text_file.write(tmp)
        bars_na.corr(numeric_only=True, method="spearman").to_csv(
            f"latex_files/{ds}_Pearson.csv", index=True
        )
        with open(f"latex_files/{ds}_table.txt", "w") as text_file:
            tmp = bars.to_latex(index=True, float_format="{:.3f}".format)
            text_file.write(tmp)
        bars.to_csv(f"latex_files/{ds}_table.csv", index=True)
