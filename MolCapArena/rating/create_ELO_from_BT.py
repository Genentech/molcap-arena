from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

from collections import defaultdict

import numpy as np

import math

from tqdm import tqdm

import itertools
import random


def get_battles(error_dict, models, THRESH=0.0):

    model_to_id = {m: i for i, m in enumerate(models)}

    models = pd.Series(models)

    num_models = len(models)

    # confusion_matrix = np.zeros((num_models, num_models))

    battle_rows = []

    for smi in error_dict:
        for inst, inst2 in itertools.combinations(error_dict[smi], 2):

            e1, e2 = inst[0], inst2[0]
            m1, m2 = inst[2], inst2[2]
            if e2 - e1 > THRESH:
                # confusion_matrix[model_to_id[m1], model_to_id[m2]] += 1
                # the algorithm breaks if a model is only ever model_a, randomize which side
                if random.random() > 0.5:
                    battle_rows.append((smi, m1, m2, e1, e2, "model_a"))
                else:
                    battle_rows.append((smi, m2, m1, e2, e1, "model_b"))
            elif e1 - e2 > THRESH:  # e2 < e1:
                # confusion_matrix[model_to_id[m2], model_to_id[m1]] += 1

                if random.random() > 0.5:
                    battle_rows.append((smi, m1, m2, e1, e2, "model_b"))
                else:
                    battle_rows.append((smi, m2, m1, e2, e1, "model_a"))

    battles = pd.DataFrame(
        battle_rows,
        columns=["smiles", "model_a", "model_b", "error_a", "error_b", "winner"],
    )
    return battles


def get_ratings(battles):

    def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None):
        from sklearn.linear_model import LogisticRegression

        ptbl_a_win = pd.pivot_table(
            df[df["winner"] == "model_a"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        # if no tie, create a zero matrix
        if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
            ptbl_tie = pd.DataFrame(
                0, index=ptbl_a_win.index, columns=ptbl_a_win.columns
            )
        else:
            ptbl_tie = pd.pivot_table(
                df[df["winner"].isin(["tie", "tie (bothbad)"])],
                index="model_a",
                columns="model_b",
                aggfunc="size",
                fill_value=0,
            )
            ptbl_tie = ptbl_tie + ptbl_tie.T

        ptbl_b_win = pd.pivot_table(
            df[df["winner"] == "model_b"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )

        # some added code to deal with 100% winrate model:
        models = set(ptbl_a_win.columns).union(ptbl_b_win.columns)
        for model in models:
            if model not in ptbl_a_win.columns:
                ptbl_a_win[model] = 0
            if model not in ptbl_b_win.index:
                new_row = pd.DataFrame(
                    [[0] * len(ptbl_b_win.columns)],
                    columns=ptbl_b_win.columns,
                    index=[model],
                )
                ptbl_b_win = pd.concat([ptbl_b_win, new_row])

        ptbl_win = (ptbl_a_win.add(ptbl_b_win.T, fill_value=0) * 2).add(
            ptbl_tie, fill_value=0
        )

        models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

        p = len(models)
        X = np.zeros([p * (p - 1) * 2, p])
        Y = np.zeros(p * (p - 1) * 2)

        cur_row = 0
        sample_weights = []
        for m_a in ptbl_win.index:
            for m_b in ptbl_win.columns:
                if m_a == m_b:
                    continue
                # if nan skip
                if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(
                    ptbl_win.loc[m_b, m_a]
                ):
                    continue
                X[cur_row, models[m_a]] = +math.log(BASE)
                X[cur_row, models[m_b]] = -math.log(BASE)
                Y[cur_row] = 1.0
                sample_weights.append(ptbl_win.loc[m_a, m_b])

                X[cur_row + 1, models[m_a]] = math.log(BASE)
                X[cur_row + 1, models[m_b]] = -math.log(BASE)
                Y[cur_row + 1] = 0.0
                sample_weights.append(ptbl_win.loc[m_b, m_a])
                cur_row += 2
        X = X[:cur_row]
        Y = Y[:cur_row]

        lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
        lr.fit(X, Y, sample_weight=sample_weights)
        elo_scores = SCALE * lr.coef_[0] + INIT_RATING

        return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

    if False:
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            elo_mle_ratings = compute_mle_elo(battles)
            print(dataset)
            print(elo_mle_ratings)
            return elo_mle_ratings
    if False:
        sf = compute_mle_elo(battles)

        rv = pd.DataFrame({"model": sf.index, "rating": sf.values}).sort_values(
            "rating", ascending=False
        )
        return rv

    BOOTSTRAP_ROUNDS = 10

    def sample_battle_even(battles, n_per_battle):
        groups = battles.groupby(["Dataset"], as_index=False)

        resampled = (
            groups[battles.columns]
            .apply(lambda grp: grp.sample(n_per_battle, replace=True))
            .reset_index(drop=True)
        )
        return resampled

    num_samples = 250000

    # Sampling Battles Evenly
    def get_bootstrap_even_sample(
        battles, n_per_battle, func_compute_elo, num_round=BOOTSTRAP_ROUNDS
    ):
        rows = []
        for n in tqdm(range(num_round), desc="sampling battles evenly"):
            resampled = sample_battle_even(battles, n_per_battle)
            rows.append(func_compute_elo(resampled))
        df = pd.DataFrame(rows)

        return df[df.median().sort_values(ascending=False).index]

    bootstrap_even_lu = get_bootstrap_even_sample(
        battles, num_samples, compute_mle_elo, num_round=BOOTSTRAP_ROUNDS
    )

    def get_error_bars(df):
        bars = (
            pd.DataFrame(
                dict(
                    lower=df.quantile(0.025),
                    rating=df.quantile(0.5),
                    upper=df.quantile(0.975),
                )
            )
            .reset_index(names="model")
            .sort_values("rating", ascending=False)
        )
        bars["error_y"] = bars["upper"] - bars["rating"]
        bars["error_y_minus"] = bars["rating"] - bars["lower"]
        bars["rating_rounded"] = np.round(bars["rating"], 2)
        return bars

    bars = get_error_bars(bootstrap_even_lu)

    return bars


if __name__ == "__main__":

    datasets = ["BACE", "ClinTox", "BBBP"]

    for dataset in datasets + ["Overall"]:
        act_dict = {}
        error_dict = defaultdict(list)

        avg_error = defaultdict(list)

        if dataset == "Overall":
            for da in datasets:
                for i in [0, 1, 2, 3, 4]:
                    df = pd.read_csv(f"test_error_data/{da}_{i}.csv")

                    df["Error"] = (df["Activity"] - df["Prediction"]) * df[
                        "Activity"
                    ] + (1 - df["Activity"]) * (df["Prediction"] - df["Activity"])

                    for row in df.iterrows():
                        d = row[1]

                        act_dict[d["SMILES"]] = d["Activity"]
                        error_dict[d["SMILES"]].append(
                            (d["Error"], d["Caption"], d["Source Model"])
                        )

                        avg_error[d["Source Model"]].append(d["Error"])

        else:
            for i in [0, 1, 2, 3, 4]:
                df = pd.read_csv(f"test_error_data/{dataset}_{i}.csv")

                df["Error"] = (df["Activity"] - df["Prediction"]) * df["Activity"] + (
                    1 - df["Activity"]
                ) * (df["Prediction"] - df["Activity"])

                for row in df.iterrows():
                    d = row[1]

                    act_dict[d["SMILES"]] = d["Activity"]
                    error_dict[d["SMILES"]].append(
                        (d["Error"], d["Caption"], d["Source Model"])
                    )

                    avg_error[d["Source Model"]].append(d["Error"])

        models = list(avg_error.keys())

        bars = get_ratings(error_dict, models)
        bars = bars.set_index("model")

        print(dataset)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(bars)
