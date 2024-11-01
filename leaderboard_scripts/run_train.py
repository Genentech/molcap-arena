import argparse

import subprocess

import os.path as osp
import sys
import random
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_exp(
    caption_sources,
    fold_idx=0,
    dataset="BBBP",
    max_epochs=25,
    validate_every_n=0,
    loss="BCE",
):

    if not (
        osp.exists(
            f"ckpts/evaluation/{dataset}/{fold_idx}/GNN/best_val_checkpoint.ckpt"
        )
        or osp.exists(f"embeddings/{dataset}_GNN_{fold_idx}_train.pkl")
    ):
        print("Starting GNN Training")
        subprocess.call(
            [
                "python",
                "MolCapArena/scripts/main.py",
                "--batch_size=32",
                "--validate_every_n=0",
                f"--max_epochs=200",
                "--num_warmup_steps=100",
                f"--data_module={dataset}",
                f"--ckpt_path=ckpts/evaluation/{dataset}/{fold_idx}/GNN/",
                f"--loss={loss}",
                "--model=GNN",
                f"--fold_idx={fold_idx}",
            ]
        )
        print("Finished GNN Training")

    print("Starting Caption Training")
    for cs in caption_sources:
        print(cs)
        sys.stdout.flush()

        # experiments = open("experiments.txt").read().splitlines()

        # print('exp:', experiments)

        # experiment_id = 'Caption' + '::' + f"Cap{dataset}" + ':'+str(fold_idx) + ':' + cs
        # print('exp_id:', experiment_id)
        if not (
            osp.exists(
                f"ckpts/evaluation/{dataset}/{fold_idx}/Caption/{cs}/best_val_checkpoint.ckpt"
            )
            or osp.exists(f"embeddings/{dataset}_{cs}_{fold_idx}_train.pkl")
        ):
            # if not experiment_id in experiments:
            # if not osp.exists(f"ckpts/evaluation/{dataset}/{fold_idx}/Caption/{cs}/best_val_checkpoint.ckpt") and not experiment_id in experiments:
            print("\t" + "Training Caption")
            rv = subprocess.call(
                [
                    "python",
                    "MolCapArena/scripts/main.py",
                    "--batch_size=32",
                    f"--validate_every_n={validate_every_n}",
                    f"--max_epochs={max_epochs}",
                    "--num_warmup_steps=100",
                    f"--data_module=Cap{dataset}",
                    f"--ckpt_path=ckpts/evaluation/{dataset}/{fold_idx}/Caption/{cs}",
                    f"--loss={loss}",
                    "--model=Caption",
                    f"--fold_idx={fold_idx}",
                    f"--caption_source={cs}",
                ]
            )

        sys.stdout.flush()
    print("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an Experiment")
    parser.add_argument("--dataset", type=str, default="BBBP")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--validate_every_n", type=int, default=0)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--specific_caption", type=str, default=None)
    args = parser.parse_args()

    dataset_to_task = {
        "BBBP": "BC",
        "BACE": "BC",
        "ClinTox": "BC",
        "ESOL": "R",
        "FreeSolv": "R",
        "Lipo": "R",
    }

    if args.loss is None:
        if dataset_to_task[args.dataset] == "BC":
            args.loss = "BCE"
        elif dataset_to_task[args.dataset] == "R":
            args.loss = "MSE"

    if args.specific_caption == None:
        import glob

        cap_files = glob.glob("captions/*.csv")

        caption_sources = [cf.split("/")[-1].split(".csv")[0] for cf in cap_files]
        random.shuffle(caption_sources)
    else:
        caption_sources = [args.specific_caption]

    run_exp(
        fold_idx=args.fold_idx,
        dataset=args.dataset,
        max_epochs=args.max_epochs,
        validate_every_n=args.validate_every_n,
        caption_sources=caption_sources,
        loss=args.loss,
    )
