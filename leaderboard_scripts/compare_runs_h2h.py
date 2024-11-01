from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

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

import glob

cap_files = glob.glob("../captions/*.csv")
caption_sources = [cf.split("/")[-1].split(".csv")[0] for cf in cap_files]  # + ['GNN']


def get_metrics():
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

                act_dict = {}
                error_dict = defaultdict(list)

                avg_error = defaultdict(list)

                df = pd.read_csv(f"{data_folder}/{method}/{dataset}/{i}/{cs}.csv")

                if task == "BC":
                    df["Error"] = (df["Activity"] - df["Prediction"]) * df[
                        "Activity"
                    ] + (1 - df["Activity"]) * (df["Prediction"] - df["Activity"])
                elif task == "R":
                    df["Error"] = np.absolute((df["Activity"] - df["Prediction"]))

                for row in df.iterrows():
                    d = row[1]

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

                if task == "BC":
                    for sm in sorted(error_dict.keys()):

                        pred = [a[0] for a in error_dict[sm]]
                        gt = [a[1] for a in error_dict[sm]]

                        rocauc = roc_auc_score(gt, pred)

                        AUCs[sm].append(rocauc)

                    for sm in sorted(error_dict.keys()):

                        pred = [a[0] for a in error_dict[sm]]
                        gt = [a[1] for a in error_dict[sm]]

                        ap = average_precision_score(gt, pred)

                        APs[sm].append(ap)

                    for sm in sorted(error_dict.keys()):

                        pred = [a[0] for a in error_dict[sm]]
                        gt = [a[1] for a in error_dict[sm]]

                        l = log_loss(gt, pred)

                        Ls[sm].append(l)

                elif task == "R":
                    for sm in sorted(error_dict.keys()):

                        pred = [a[0] for a in error_dict[sm]]
                        gt = [a[1] for a in error_dict[sm]]

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

                    e = [a[4] for a in error_dict[sm]]

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
        from create_ELO_from_BT import get_ratings
        from get_ELO_head2head import get_battles_h2h

        battles = get_battles_h2h(data_folder, "SVM", datasets, splits)
        bars = get_ratings(battles)

        bars["ROC-AUC"] = bars["model"].map(AUCs)
        bars["Loss"] = bars["model"].map(Ls)
        bars["Avg. BCE Error"] = bars["model"].map(AEs)
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
            print(bars_na)

        return bars_na


if True:
    datasets = ["BBBP", "BACE", "ClinTox"]
    splits = [0, 1, 2, 3, 4]
    method = "SVM"
    data_folder = "battles"

    df1 = get_metrics()
    datasets1 = datasets

    datasets = ["ESOL", "FreeSolv", "Lipo"]
    splits = [0, 1, 2, 3, 4]
    method = "SVM"
    data_folder = "battles"
    datasets2 = datasets

    df2 = get_metrics()


# Step 1: Align DataFrames based on index
aligned_df1, aligned_df2 = df1.align(df2, join="inner", axis=0)

# Step 2: Concatenate DataFrames along columns
combined_df = pd.concat([aligned_df1, aligned_df2], axis=1, keys=["df1", "df2"])

ds1 = str(datasets1)
ds2 = str(datasets2)

# Step 3: Calculate the correlation matrix
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "expand_frame_repr", False
):
    print("Pearson:")
    print(combined_df.corr())

    with open(f"latex_files/{ds1}_vs_{ds2}_splits_pearson.txt", "w") as text_file:
        tmp = combined_df.corr().to_latex(index=True, float_format="{:.3f}".format)
        text_file.write(tmp)

    print("Spearman:")
    print(combined_df.corr(method="spearman"))

    with open(f"latex_files/{ds1}_vs_{ds2}_splits_spearman.txt", "w") as text_file:
        tmp = combined_df.corr(method="spearman").to_latex(
            index=True, float_format="{:.3f}".format
        )
        text_file.write(tmp)
