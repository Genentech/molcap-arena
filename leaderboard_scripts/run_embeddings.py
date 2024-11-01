import argparse

import subprocess

import os.path as osp
import sys
import random
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_exp(caption_sources, fold_idx=0, dataset="BBBP", validate_every_n=0):

    # experiments = open("experiments_emb.txt").read().splitlines()
    # experiment_id = 'GNN' + ':' + f"{dataset}" + ':'+str(fold_idx) + ':' + 'GNN'
    if not osp.exists(f"embeddings/{dataset}_GNN_{fold_idx}_train.pkl"):
        # if not experiment_id in experiments:
        # with open("experiments_emb.txt", "a") as myfile:
        #    myfile.write(experiment_id + '\n')

        subprocess.call(
            [
                "python",
                "MolCapArena/scripts/build_embeddings.py",
                "--batch_size=128",
                f"--validate_every_n={validate_every_n}",
                "--max_epochs=0",
                "--num_warmup_steps=100 ",
                f"--data_module={dataset}",
                "--loss=BCE",
                "--model=GNN",
                f"--fold_idx={fold_idx}",
                f"--GNN_checkpoint=ckpts/evaluation/{dataset}/{fold_idx}/GNN/best_val_checkpoint.ckpt",
                f"--caption_source=GNN",
            ]
        )

    for cs in caption_sources:
        print(cs)
        sys.stdout.flush()

        # experiments = open("experiments_emb.txt").read().splitlines()
        # experiment_id = 'Caption' + ':' + f"Cap{dataset}" + ':'+str(fold_idx) + ':' + cs
        if not osp.exists(f"embeddings/{dataset}_{cs}_{fold_idx}_train.pkl"):
            # with open("experiments_emb.txt", "a") as myfile:
            #    myfile.write(experiment_id + '\n')

            subprocess.call(
                [
                    "python",
                    "MolCapArena/scripts/build_embeddings.py",
                    "--batch_size=128",
                    f"--validate_every_n={validate_every_n}",
                    "--max_epochs=0",
                    "--num_warmup_steps=100 ",
                    f"--data_module=Cap{dataset}",
                    "--loss=BCE",
                    "--model=Caption",
                    f"--fold_idx={fold_idx}",
                    f"--resume_from_checkpoint=ckpts/evaluation/{dataset}/{fold_idx}/Caption/{cs}/best_val_checkpoint.ckpt",
                    f"--caption_source={cs}",
                ]
            )
        sys.stdout.flush()
    print("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an Experiment")
    parser.add_argument("--dataset", type=str, default="BBBP")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--validate_every_n", type=int, default=0)
    parser.add_argument("--specific_caption", type=str, default=None)
    args = parser.parse_args()

    import glob

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
        validate_every_n=args.validate_every_n,
        caption_sources=caption_sources,
    )
