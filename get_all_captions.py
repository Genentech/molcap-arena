from rdkit import Chem
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import time

import argparse

import selfies as sf

import sys
import os

import subprocess


def canonicalize(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


def to_sf(smi):
    if "*" in smi:
        return sf.encoder(smi.replace("*", "[Rn]"), strict=False).replace("[Rn]", "[*]")
    return sf.encoder(smi, strict=False)


parser = argparse.ArgumentParser(description="Captioner")
parser.add_argument("--model", default="", type=str)
parser.add_argument("--debug", default="", action=argparse.BooleanOptionalAction)


args = parser.parse_args()

if "Frags" in args.model:
    source_data = "all_smiles_BRICS.csv"
else:
    source_data = "all_smiles.csv"

print(args.model)
print()
subprocess.run(["nvidia-smi"])

if args.debug:
    df = pd.read_csv(source_data)[:20]
else:
    df = pd.read_csv(source_data)

batch_size = 4  # default, not used by most captioners

df["SMILES"] = df["SMILES"].map(canonicalize)
df["SELFIES"] = df["SMILES"].map(to_sf)

if args.model == "MolT5":
    from captioners import MolT5_captioner

    df, name = MolT5_captioner.create_captions(df, batch_size)
if args.model == "MolT5_LPM24":
    from captioners import MolT5_LPM24_captioner

    df, name = MolT5_LPM24_captioner.create_captions(df, batch_size)
if args.model == "TextChemT5":
    from captioners import TextChemT5_captioner

    df, name = TextChemT5_captioner.create_captions(df, batch_size)
elif args.model == "BioT5":
    from captioners import BioT5_captioner

    df, name = BioT5_captioner.create_captions(df, batch_size)
elif args.model == "LlaSMol":
    from captioners import LlaSMol_captioner

    df, name = LlaSMol_captioner.create_captions(df, batch_size)


elif args.model == "Llama3-8B-Generic":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
    )
elif args.model == "Llama3-8B-Drug":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    )
elif args.model == "Llama3-8B-Bio":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecule has important biological properties. Tell me biological properties of the molecule and relevant functionality and applications.",
    )
elif args.model == "Llama3-8B-Chem":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important chemical properties. Tell me chemical properties of the molecule and relevant functionality and applications.",
    )
elif args.model == "Llama3-8B-Quant":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important quantum properties. Tell me quantum properties of the molecule and relevant functionality and applications.",
    )


elif args.model == "Llama3-8B-Frags-Generic":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
    )
elif args.model == "Llama3-8B-Frags-Drug":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecular fragments constitute a possible drug. Tell me properties of these fragments, such as the mechanism of action, class, and target, which might be relevant.",
    )
elif args.model == "Llama3-8B-Frags-Bio":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecular fragments have important biological properties. Tell me biological properties of these fragments and possible relevant functionality and applications.",
    )
elif args.model == "Llama3-8B-Frags-Chem":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important chemical properties. Tell me chemical properties of these fragments and possible relevant functionality and applications.",
    )
elif args.model == "Llama3-8B-Frags-Quant":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important quantum properties. Tell me quantum properties of these fragments and possible relevant functionality and applications.",
    )


elif args.model == "Llama3-70B-Generic":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
    )
elif args.model == "Llama3-70B-Drug":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    )
elif args.model == "Llama3-70B-Bio":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecule has important biological properties. Tell me biological properties of the molecule and relevant functionality and applications.",
    )
elif args.model == "Llama3-70B-Chem":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important chemical properties. Tell me chemical properties of the molecule and relevant functionality and applications.",
    )
elif args.model == "Llama3-70B-Quant":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important quantum properties. Tell me quantum properties of the molecule and relevant functionality and applications.",
    )


elif args.model == "Llama3-70B-Frags-Generic":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
    )
elif args.model == "Llama3-70B-Frags-Drug":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecular fragments constitute a possible drug. Tell me properties of these fragments, such as the mechanism of action, class, and target, which might be relevant.",
    )
elif args.model == "Llama3-70B-Frags-Bio":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecular fragments have important biological properties. Tell me biological properties of these fragments and possible relevant functionality and applications.",
    )
elif args.model == "Llama3-70B-Frags-Chem":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important chemical properties. Tell me chemical properties of these fragments and possible relevant functionality and applications.",
    )
elif args.model == "Llama3-70B-Frags-Quant":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important quantum properties. Tell me quantum properties of these fragments and possible relevant functionality and applications.",
    )


elif args.model == "MolInstructions_molecule":
    from captioners import MolInstructions_captioner

    df, name = MolInstructions_captioner.create_captions(
        df,
        batch_size,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_name="zjunlp/llama3-instruct-molinst-molecule-8b",
    )

elif args.model == "ChemLLM":  # batch size has to be 1 b/c of rotary position bug
    from captioners import ChemLLM_captioner

    df, name = ChemLLM_captioner.create_captions(
        df,
    )

elif args.model == "ChemDFM":
    from captioners import ChemDFM_captioner

    df, name = ChemDFM_captioner.create_captions(
        df,
        batch_size,
    )

elif args.model == "BioMedGPT":
    sys.path.append("./captioners/OpenBioMed/")
    sys.path.append("./captioners/OpenBioMed/open_biomed/")
    from captioners import BioMedGPT_captioner

    df, name = BioMedGPT_captioner.create_captions(
        df,
    )
elif args.model == "3D-MoLM":
    sys.path.append("./captioners/ThreeD_MoLM/")
    sys.path.append("./captioners/ThreeD_MoLM/model/")
    from captioners import ThreeD_MoLM_captioner

    df, name = ThreeD_MoLM_captioner.create_captions(
        df,
    )

elif args.model == "BioT5_plus":
    from captioners import BioT5_plus_captioner

    df, name = BioT5_plus_captioner.create_captions(df, batch_size)


elif args.model == "GPT-4o-Drug":
    from captioners import OpenAI_captioner

    df, name = OpenAI_captioner.create_captions(
        df,
        openai_api_key=os.environ["API_KEY_ES2"],
        openai_api_base=os.environ["API_BASE"],
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecule might be a drug. Analyze the structure of the molecule to tell me its properties, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="gpt-4o",
        dump_file="GPT4o_drug_backup.json",
    )


elif args.model == "Gemma2-9B-Generic":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        2,
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Drug":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        2,
        prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Bio":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        2,
        prompt="The following molecule has important biological properties. Tell me biological properties of the molecule and relevant functionality and applications.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Chem":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        2,
        prompt="The following molecule has important chemical properties. Tell me chemical properties of the molecule and relevant functionality and applications.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Quant":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        2,
        prompt="The following molecule has important quantum properties. Tell me quantum properties of the molecule and relevant functionality and applications.",
        model_name="google/gemma-2-9b-it",
    )


elif args.model == "Gemma2-27B-Generic":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        1,
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Drug":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        1,
        prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Bio":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        1,
        prompt="The following molecule has important biological properties. Tell me biological properties of the molecule and relevant functionality and applications.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Chem":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        1,
        prompt="The following molecule has important chemical properties. Tell me chemical properties of the molecule and relevant functionality and applications.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Quant":
    from captioners import Gemma2_captioner

    df, name = Gemma2_captioner.create_captions(
        df,
        1,
        prompt="The following molecule has important quantum properties. Tell me quantum properties of the molecule and relevant functionality and applications.",
        model_name="google/gemma-2-27b-it",
    )


elif args.model == "MistralNeMo-12B-Generic":
    from captioners import Mistral_NeMo_12B_captioner

    df, name = Mistral_NeMo_12B_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
    )
elif args.model == "MistralNeMo-12B-Drug":
    from captioners import Mistral_NeMo_12B_captioner

    df, name = Mistral_NeMo_12B_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    )
elif args.model == "MistralNeMo-12B-Bio":
    from captioners import Mistral_NeMo_12B_captioner

    df, name = Mistral_NeMo_12B_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecule has important biological properties. Tell me biological properties of the molecule and relevant functionality and applications.",
    )
elif args.model == "MistralNeMo-12B-Chem":
    from captioners import Mistral_NeMo_12B_captioner

    df, name = Mistral_NeMo_12B_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important chemical properties. Tell me chemical properties of the molecule and relevant functionality and applications.",
    )
elif args.model == "MistralNeMo-12B-Quant":
    from captioners import Mistral_NeMo_12B_captioner

    df, name = Mistral_NeMo_12B_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important quantum properties. Tell me quantum properties of the molecule and relevant functionality and applications.",
    )


elif args.model == "Llama3.1-8B-Generic":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Drug":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Bio":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecule has important biological properties. Tell me biological properties of the molecule and relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Chem":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important chemical properties. Tell me chemical properties of the molecule and relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Quant":
    from captioners import Llama3_8B_captioner

    df, name = Llama3_8B_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important quantum properties. Tell me quantum properties of the molecule and relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )


elif args.model == "Llama3.1-8B-Frags-Generic":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Frags-Drug":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecular fragments constitute a possible drug. Tell me properties of these fragments, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Frags-Bio":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecular fragments have important biological properties. Tell me biological properties of these fragments and possible relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Frags-Chem":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important chemical properties. Tell me chemical properties of these fragments and possible relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
elif args.model == "Llama3.1-8B-Frags-Quant":
    from captioners import Llama3_8B_frag_captioner

    df, name = Llama3_8B_frag_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important quantum properties. Tell me quantum properties of these fragments and possible relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )


elif args.model == "Llama3.1-70B-Generic":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Drug":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Bio":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecule has important biological properties. Tell me biological properties of the molecule and relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Chem":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important chemical properties. Tell me chemical properties of the molecule and relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Quant":
    from captioners import Llama3_70B_captioner

    df, name = Llama3_70B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecule has important quantum properties. Tell me quantum properties of the molecule and relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )


elif args.model == "Llama3.1-70B-Frags-Generic":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Frags-Drug":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecular fragments constitute a possible drug. Tell me properties of these fragments, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Frags-Bio":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecular fragments have important biological properties. Tell me biological properties of these fragments and possible relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Frags-Chem":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important chemical properties. Tell me chemical properties of these fragments and possible relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )
elif args.model == "Llama3.1-70B-Frags-Quant":
    from captioners import Llama3_70B_frag_captioner

    df, name = Llama3_70B_frag_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important quantum properties. Tell me quantum properties of these fragments and possible relevant functionality and applications.",
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    )

elif args.model == "BlankCaption":
    from captioners import Blank_captioner

    df, name = Blank_captioner.create_captions(
        df,
        64,
    )


elif args.model == "Gemma2-9B-Frags-Generic":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        2,
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Frags-Drug":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        2,
        prompt="The following molecular fragments constitute a possible drug. Tell me properties of these fragments, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Frags-Bio":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        2,
        prompt="The following molecular fragments have important biological properties. Tell me biological properties of these fragments and possible relevant functionality and applications.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Frags-Chem":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        2,
        prompt="The following molecular fragments have important chemical properties. Tell me chemical properties of these fragments and possible relevant functionality and applications.",
        model_name="google/gemma-2-9b-it",
    )
elif args.model == "Gemma2-9B-Frags-Quant":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        2,
        prompt="The following molecular fragments have important quantum properties. Tell me quantum properties of these fragments and possible relevant functionality and applications.",
        model_name="google/gemma-2-9b-it",
    )


elif args.model == "Gemma2-27B-Frags-Generic":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        1,
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Frags-Drug":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        1,
        prompt="The following molecular fragments constitute a possible drug. Tell me properties of these fragments, such as the mechanism of action, class, and target, which might be relevant.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Frags-Bio":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        1,
        prompt="The following molecular fragments have important biological properties. Tell me biological properties of these fragments and possible relevant functionality and applications.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Frags-Chem":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        1,
        prompt="The following molecular fragments have important chemical properties. Tell me chemical properties of these fragments and possible relevant functionality and applications.",
        model_name="google/gemma-2-27b-it",
    )
elif args.model == "Gemma2-27B-Frags-Quant":
    from captioners import Gemma2_frag_captioner

    df, name = Gemma2_frag_captioner.create_captions(
        df,
        1,
        prompt="The following molecular fragments have important quantum properties. Tell me quantum properties of these fragments and possible relevant functionality and applications.",
        model_name="google/gemma-2-27b-it",
    )


elif args.model == "MistralNeMo-12B-Frags-Generic":
    from captioners import Mistral_NeMo_12B_frag_captioner

    df, name = Mistral_NeMo_12B_frag_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
    )
elif args.model == "MistralNeMo-12B-Frags-Drug":
    from captioners import Mistral_NeMo_12B_frag_captioner

    df, name = Mistral_NeMo_12B_frag_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in medicinal chemistry. In particular, you specialize in the design of small molecule drugs. You are sharing your knowledge of known chemistry with a colleague seeking to better understand drug design.",
        prompt="The following molecular fragments constitute a possible drug. Tell me properties of these fragments, such as the mechanism of action, class, and target, which might be relevant.",
    )
elif args.model == "MistralNeMo-12B-Frags-Bio":
    from captioners import Mistral_NeMo_12B_frag_captioner

    df, name = Mistral_NeMo_12B_frag_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in biology. You are sharing your knowledge of known biochemistry with a colleague.",
        prompt="The following molecular fragments have important biological properties. Tell me biological properties of these fragments and possible relevant functionality and applications.",
    )
elif args.model == "MistralNeMo-12B-Frags-Chem":
    from captioners import Mistral_NeMo_12B_frag_captioner

    df, name = Mistral_NeMo_12B_frag_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important chemical properties. Tell me chemical properties of these fragments and possible relevant functionality and applications.",
    )
elif args.model == "MistralNeMo-12B-Frags-Quant":
    from captioners import Mistral_NeMo_12B_frag_captioner

    df, name = Mistral_NeMo_12B_frag_captioner.create_captions(
        df,
        24,
        sys_prompt="You are a helpful expert in quantum chemistry. You are sharing your knowledge of known chemistry with a colleague.",
        prompt="The following molecular fragments have important quantum properties. Tell me quantum properties of these fragments and possible relevant functionality and applications.",
    )


elif args.model == "GPT-4o-Generic":
    from captioners import OpenAI_captioner

    df, name = OpenAI_captioner.create_captions(
        df,
        openai_api_key=os.environ["API_KEY_ES2"],
        openai_api_base=os.environ["API_BASE"],
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
        model_name="gpt-4o",
        dump_file="gpt4_backups/GPT4o_generic_backup.json",
    )


elif args.model == "GPT-4o-Frags-Generic":
    from captioners import OpenAI_frag_captioner

    df, name = OpenAI_frag_captioner.create_captions(
        df,
        openai_api_key=os.environ["API_KEY_ES2"],
        openai_api_base=os.environ["API_BASE"],
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
        model_name="gpt-4o",
        dump_file="gpt4_backups/GPT4o_generic_frag_backup.json",
    )


elif args.model == "Llama3.1-405B-Generic":
    from captioners import Llama3_405B_captioner

    df, name = Llama3_405B_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
        model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    )

elif args.model == "Llama3.1-405B-Frags-Generic":
    from captioners import Llama3_405B_frag_captioner

    df, name = Llama3_405B_frag_captioner.create_captions(
        df,
        2,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecular fragments have important properties. Tell me about relevant functionality and applications of these molecular fragments.",
        model_name="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    )


elif args.model == "BBBP_captioner":
    from captioners import BBBP_captioner

    df, name = BBBP_captioner.create_captions(
        df,
        batch_size,
    )


elif args.model == "Llama3-8B-Task":
    from captioners import Llama3_8B_task_captioner

    task_desc = {
        "BBBP": "blood brain barrier penetration prediction",
        "ClinTox": "clinical trial toxicity prediction",
        "BACE": "beta-secretase 1 binding prediction",
        "ESOL": "aqueous solubility prediction",
        "FreeSolv": "hydration-free energy prediction of small molecules in water",
        "Lipo": "lipophilicity prediction",
    }
    df, name = Llama3_8B_task_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties.\nMolecule: {}\n\nTell me about relevant properties and functionality of this molecule which indicate whether it can complete the following task.\nTask: {}\n\n",
        negative=False,
        task_desc=task_desc,
    )


elif args.model == "YOUR_NAME":
    from captioners import YOUR_NAME_captioner

    df, name = YOUR_NAME_captioner.create_captions(
        df,
        batch_size,
        sys_prompt="You are a helpful expert in chemistry. You are sharing your knowledge of known chemistry with a colleague seeking to better understand how small molecules work.",
        prompt="The following molecule has important properties. Tell me about relevant functionality and applications of this molecule.",
    )

df.to_csv(f"captions/{args.model}.csv", index=False)
