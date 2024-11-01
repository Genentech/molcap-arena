import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import pandas as pd

from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose


def canonicalize(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


source_data = "all_smiles.csv"

df = pd.read_csv(source_data)

df["SMILES"] = df["SMILES"].map(canonicalize)


bad_mols = set(
    [
        "CC[C@H](C)[C@@H]1NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](C)NC(=O)[C@H](CCCCN)NC(=O)[C@@H]2CSSC[C@@H]3NC(=O)CNC(=O)CNC(=O)[C@H](Cc4ccc(O)cc4)NC(=O)[C@H](C(C)C)NC(=O)[C@H](Cc4ccccc4)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CSSC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](CO)NC(=O)[C@H](CCCCN)NC(=O)[C@H](Cc4ccccc4)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCCCN)NC(=O)[C@H](C)NC(=O)[C@H](CCCNC(=N)N)NC3=O)C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@H](C(=O)NCC(=O)NCC(=O)N[C@@H](C)C(=O)O)CSSC[C@H](NC(=O)[C@H](Cc3ccccc3)NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H]3CCCN3C(=O)[C@@H](N)CCCNC(=N)N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(=O)O)C(=O)N3CCC[C@H]3C(=O)N3CCC[C@H]3C(=O)N[C@@H](Cc3ccc(O)cc3)C(=O)N[C@@H]([C@@H](C)O)C(=O)NCC(=O)N3CCC[C@H]3C(=O)N2)NC(=O)[C@H](CC(C)C)NC(=O)CNC(=O)[C@H](C)NC(=O)[C@H](CCCCN)NC(=O)[C@H](C)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H]([C@@H](C)CC)NC1=O",
        "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CS)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CS)NC(=O)[C@@H]1CCCN1C(=O)CNC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CC(=O)O)NC(=O)[C@H](C)NC(=O)[C@H](CCCCN)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](C)NC(=O)[C@H](CS)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CO)NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@H](CCSC)NC(=O)[C@H](C)NC(=O)[C@@H](N)CCC(=O)O)[C@@H](C)CC)[C@@H](C)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)NCC(=O)NCC(=O)N[C@@H](CS)C(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CS)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CS)C(=O)N[C@H](C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(=O)O)C(=O)O)[C@@H](C)O",
        "CC[C@H](C)[C@H](NC(=O)CN)C(=O)N[C@H](C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@H]1CSSC[C@@H]2NC(=O)[C@H]([C@@H](C)CC)NC(=O)[C@H](CO)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CSSC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc3cnc[nH]3)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)Cc3ccccc3)C(C)C)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](Cc3cnc[nH]3)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc3ccc(O)cc3)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@H](C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](Cc3ccccc3)C(=O)N[C@@H](Cc3ccccc3)C(=O)N[C@@H](Cc3ccc(O)cc3)C(=O)N[C@H](C(=O)N3CCC[C@H]3C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)O)[C@@H](C)O)[C@@H](C)O)CSSC[C@@H](C(=O)N[C@@H](CC(N)=O)C(=O)O)NC(=O)[C@H](Cc3ccc(O)cc3)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](Cc3ccc(O)cc3)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CO)NC2=O)NC1=O)C(C)C",
        "CC[C@H](C)[C@@H]1NC(=O)[C@@H]2CSSC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@@H]3CSSC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](NC(=O)[C@H](Cc4ccc(O)cc4)NC(=O)[C@@H](NC(=O)[C@@H](N)C(C)C)C(C)C)[C@@H](C)O)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N3)C(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CSSC[C@@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)N[C@H](C(=O)N3CCC[C@H]3C(=O)N[C@@H](CCCCN)C(=O)N3CCC[C@H]3C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc3cnc[nH]3)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc3ccccc3)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@H](C(=O)N3CCC[C@H]3C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc3ccc(O)cc3)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(N)=O)C(=O)O)[C@@H](C)CC)[C@@H](C)O)[C@@H](C)O)C(C)C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCC(=O)O)NC(=O)CNC(=O)[C@H](CC(=O)O)NC(=O)[C@H](CO)NC(=O)CNC(=O)[C@H](CC(C)C)NC1=O)C(=O)NCC(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCCCN)C(=O)N2",
        "CCCCCCCCCCCCCC(=O)NCCCC[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](Cc1ccccc1)NC(=O)CNC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCC(=O)O)NC(=O)CNC(=O)[C@@H]1CSSC[C@@H](C(=O)N[C@@H](CC(N)=O)C(=O)O)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CO)NC(=O)[C@@H]2CSSC[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](NC(=O)[C@@H](NC(=O)CN)[C@@H](C)CC)C(C)C)C(=O)N[C@@H](CSSC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc3c[nH]cn3)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)Cc3ccccc3)C(C)C)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](Cc3c[nH]cn3)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc3ccc(O)cc3)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N1)C(=O)N[C@@H]([C@@H](C)O)C(=O)N[C@@H](CO)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N2)[C@@H](C)O)C(=O)O",
        "COCCO[C@@H]1[C@H](SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(=O)[nH]c3=O)[C@H](OCCOC)[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(N)nc3=O)[C@H](OCCOC)[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cnc4c(N)ncnc43)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cnc4c(=O)[nH]c(N)nc43)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(=O)[nH]c3=O)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(N)nc3=O)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(=O)[nH]c3=O)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cnc4c(=O)[nH]c(N)nc43)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(N)nc3=O)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(=O)[nH]c3=O)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(=O)[nH]c3=O)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cc(C)c(N)nc3=O)C[C@@H]2SP(=O)([O-])OC[C@H]2O[C@@H](n3cnc4c(=O)[nH]c(N)nc43)[C@H](OCCOC)[C@@H]2SP(=O)([O-])OC[C@@H]2O[C@H](n3cc(C)c(N)nc3=O)[C@@H](OCCOC)[C@H]2SP(=O)([O-])OC[C@@H]2O[C@H](n3cnc4c(N)ncnc43)[C@@H](OCCOC)[C@H]2SP(=O)([O-])OC[C@@H]2O[C@H](n3cc(C)c(N)nc3=O)[C@@H](OCCOC)[C@H]2SP(=O)([O-])OC[C@@H]2O[C@H](n3cc(C)c(N)nc3=O)[C@@H](OCCOC)[C@H]2O)[C@@H](COP(=O)([O-])S[C@H]2[C@@H](OCCOC)[C@H](n3cc(C)c(N)nc3=O)O[C@@H]2COP(=O)([O-])S[C@H]2[C@@H](OCCOC)[C@H](n3cnc4c(=O)[nH]c(N)nc43)O[C@@H]2CO)O[C@H]1n1cc(C)c(N)nc1=O.[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+].[Na+]",
        "CC[C@H](C)[C@H](NC(=O)CN)C(=O)N[C@H](C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@H]1CSSC[C@@H]2NC(=O)[C@H]([C@@H](C)CC)NC(=O)[C@H](CO)NC(=O)[C@H]([C@@H](C)O)NC(=O)[C@H](CSSC[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc3cnc[nH]3)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](NC(=O)[C@@H](N)Cc3ccccc3)C(C)C)C(=O)NCC(=O)N[C@@H](CO)C(=O)N[C@@H](Cc3cnc[nH]3)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc3ccc(O)cc3)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@H](C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](Cc3ccccc3)C(=O)N[C@@H](Cc3ccccc3)C(=O)N[C@@H](Cc3ccc(O)cc3)C(=O)N[C@H](C(=O)N3CCC[C@H]3C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O)[C@@H](C)O)[C@@H](C)O)CSSC[C@@H](C(=O)NCC(=O)O)NC(=O)[C@H](Cc3ccc(O)cc3)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](Cc3ccc(O)cc3)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CO)NC2=O)NC1=O)C(C)C",
    ]
)


def create_BRICS(smi):
    try:
        m = Chem.MolFromSmiles(smi)

        if smi in bad_mols:
            frags = {smi}
        else:
            frags = set(BRICSDecompose(m))

        return frags
    except:
        return set()


df["BRICS"] = df["SMILES"].progress_map(create_BRICS)


df.to_csv(f"all_smiles_BRICS.csv")
