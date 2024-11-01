import sys

# sys.path.append('.captioners')
# sys.path.append('.captioners/ThreeD_MoLM/model/')
# sys.path.append('.captioners/ThreeD_MoLM/')

import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

from rdkit import Chem

import time

import pandas as pd

from blip2_llama_inference import Blip2Llama
from unimol import SimpleUniMolModel
from rdkit import Chem
from rdkit.Chem import AllChem
from unicore.data import Dictionary
import numpy as np
import torch
from scipy.spatial import distance_matrix


def create_captions(df):

    tensor_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    batch_size = 1

    name = "3D-MoLM"

    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        ### models
        parser.add_argument(
            "--bert_name", type=str, default="all_checkpoints/scibert_scivocab_uncased"
        )
        parser.add_argument(
            "--llm_model", type=str, default="all_checkpoints/llama-2-7b-hf"
        )

        ### flash attention
        parser.add_argument("--enable_flash", action="store_false", default=False)

        ### lora settings
        parser.add_argument("--lora_r", type=int, default=8)
        parser.add_argument("--lora_alpha", type=int, default=32)
        parser.add_argument("--lora_dropout", type=int, default=0.1)
        parser.add_argument(
            "--lora_path",
            type=str,
            default="all_checkpoints/generalist/generalist.ckpt",
        )

        ### q-former settings
        parser.add_argument("--cross_attention_freq", type=int, default=2)
        parser.add_argument("--num_query_token", type=int, default=8)

        parser = SimpleUniMolModel.add_args(parser)

        args, unknown = parser.parse_known_args()
        return args

    args = get_args()

    model_max_length = 512

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    model = Blip2Llama(args).to(tensor_type)
    device = torch.device("cuda")
    model.to(device)
    tokenizer = model.llm_tokenizer

    def smiles2graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if (np.asarray(atoms) == "H").all():
            return None
        coordinate_list = []
        res = AllChem.EmbedMolecule(mol, maxAttempts=100)
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                pass
            coordinates = mol.GetConformer().GetPositions()
        elif res == -1:
            mol_tmp = Chem.MolFromSmiles(smiles)
            AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000)
            mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol_tmp)
            except:
                pass
            try:
                coordinates = mol_tmp.GetConformer().GetPositions()
            except:
                return None
        coordinates = coordinates.astype(np.float32)
        assert len(atoms) == len(
            coordinates
        ), "coordinates shape is not align with {}".format(smiles)
        assert coordinates.shape[1] == 3

        atoms = np.asarray(atoms)
        ## remove the hydrogen
        mask_hydrogen = atoms != "H"
        if sum(mask_hydrogen) > 0:
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

        ## atom vectors
        # dictionary = Dictionary.load('data_provider/unimol_dict.txt')
        dictionary = Dictionary.load(
            "captioners/ThreeD_MoLM/data_provider/unimol_dict.txt"
        )
        dictionary.add_symbol("[MASK]", is_special=True)
        atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()

        ## normalize coordinates:
        coordinates = coordinates - coordinates.mean(axis=0)

        ## add_special_token:
        atom_vec = torch.cat(
            [
                torch.LongTensor([dictionary.bos()]),
                atom_vec,
                torch.LongTensor([dictionary.eos()]),
            ]
        )
        coordinates = np.concatenate(
            [np.zeros((1, 3)), coordinates, np.zeros((1, 3))], axis=0
        )

        ## obtain edge types; which is defined as the combination of two atom types
        edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        coordinates, dist = torch.from_numpy(coordinates), torch.from_numpy(dist)

        return atom_vec, dist, edge_type

    def get_3d_graph(smiles=None, sdf_file=None):
        if sdf_file is not None:
            d3_graph = sdf2graph(sdf_file)
        elif smiles is not None:
            d3_graph = smiles2graph(smiles)
        else:
            raise ValueError("smiles must be provided")
        return d3_graph

    def tokenize(tokenizer, text):
        text_tokens = tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        is_mol_token = text_tokens.input_ids == tokenizer.mol_token_id
        text_tokens["is_mol_token"] = is_mol_token
        assert torch.sum(is_mol_token).item() == 8

        return text_tokens

    def build_input(smiles_input):
        graph = get_3d_graph(smiles=smiles_input, sdf_file=None)
        if graph is None:
            pass
        else:
            atom_vec, dist, edge_type = graph
            atom_vec, dist, edge_type = (
                atom_vec.unsqueeze(0),
                dist.unsqueeze(0).to(tensor_type),
                edge_type.unsqueeze(0),
            )
            atom_vec, dist, edge_type = (
                atom_vec.to(device),
                dist.to(device),
                edge_type.to(device),
            )
            graph = (atom_vec, dist, edge_type)
        prompt = (
            "Below is an instruction that describes a task, paired with an input molecule. Write a response that appropriately completes the request.\n"
            "Instruction: Describe the input molecule.\n"
            "Input molecule: {} <mol><mol><mol><mol><mol><mol><mol><mol>.\n"
            "Response: "
        )

        inp = prompt.format(smiles_input)
        return graph, inp

    def get_captions(inputs):
        all_caps = []

        for inps in tqdm(
            divide_chunks(inputs, batch_size), total=math.ceil(len(inputs) / batch_size)
        ):
            graph, inp = build_input(inps[0])

            input_tokens = tokenize(tokenizer, inp)
            input_tokens.to(device)
            if graph is not None:
                caps = model.generate(graph, input_tokens)
            else:
                caps = [""]

            all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["SMILES"].tolist())

    df["captions"] = captions

    return df, name
