from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd

from collections import defaultdict

import numpy as np


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


if "DATASETS" in os.environ:
    datasets = os.environ["DATASETS"].split()


# grouping = 'prompt_var'
# grouping = 'prompt_var_frag'
# grouping = 'size'
# grouping = 'frag'
grouping = "model"


# Grouping across prompt variations
if grouping == "model":
    groups = {
        "Llama3-8B-Generic": "Llama3-8B",
        "Llama3-8B-Drug": "Llama3-8B",
        "Llama3-8B-Bio": "Llama3-8B",
        "Llama3-8B-Chem": "Llama3-8B",
        "Llama3-8B-Quant": "Llama3-8B",
        "Llama3-8B-Frags-Generic": "Llama3-8B-Frags",
        "Llama3-8B-Frags-Drug": "Llama3-8B-Frags",
        "Llama3-8B-Frags-Bio": "Llama3-8B-Frags",
        "Llama3-8B-Frags-Chem": "Llama3-8B-Frags",
        "Llama3-8B-Frags-Quant": "Llama3-8B-Frags",
        "Llama3.1-8B-Generic": "Llama3.1-8B",
        "Llama3.1-8B-Drug": "Llama3.1-8B",
        "Llama3.1-8B-Bio": "Llama3.1-8B",
        "Llama3.1-8B-Chem": "Llama3.1-8B",
        "Llama3.1-8B-Quant": "Llama3.1-8B",
        "Llama3.1-8B-Frags-Generic": "Llama3.1-8B-Frags",
        "Llama3.1-8B-Frags-Drug": "Llama3.1-8B-Frags",
        "Llama3.1-8B-Frags-Bio": "Llama3.1-8B-Frags",
        "Llama3.1-8B-Frags-Chem": "Llama3.1-8B-Frags",
        "Llama3.1-8B-Frags-Quant": "Llama3.1-8B-Frags",
        "Llama3-70B-Generic": "Llama3-70B",
        "Llama3-70B-Drug": "Llama3-70B",
        "Llama3-70B-Bio": "Llama3-70B",
        "Llama3-70B-Chem": "Llama3-70B",
        "Llama3-70B-Quant": "Llama3-70B",
        "Llama3-70B-Frags-Generic": "Llama3-70B-Frags",
        "Llama3-70B-Frags-Drug": "Llama3-70B-Frags",
        "Llama3-70B-Frags-Bio": "Llama3-70B-Frags",
        "Llama3-70B-Frags-Chem": "Llama3-70B-Frags",
        "Llama3-70B-Frags-Quant": "Llama3-70B-Frags",
        "Llama3.1-70B-Generic": "Llama3.1-70B",
        "Llama3.1-70B-Drug": "Llama3.1-70B",
        "Llama3.1-70B-Bio": "Llama3.1-70B",
        "Llama3.1-70B-Chem": "Llama3.1-70B",
        "Llama3.1-70B-Quant": "Llama3.1-70B",
        "Llama3.1-70B-Frags-Generic": "Llama3.1-70B-Frags",
        "Llama3.1-70B-Frags-Drug": "Llama3.1-70B-Frags",
        "Llama3.1-70B-Frags-Bio": "Llama3.1-70B-Frags",
        "Llama3.1-70B-Frags-Chem": "Llama3.1-70B-Frags",
        "Llama3.1-70B-Frags-Quant": "Llama3.1-70B-Frags",
        "Gemma2-9B-Generic": "Gemma2-9B",
        "Gemma2-9B-Drug": "Gemma2-9B",
        "Gemma2-9B-Bio": "Gemma2-9B",
        "Gemma2-9B-Chem": "Gemma2-9B",
        "Gemma2-9B-Quant": "Gemma2-9B",
        "Gemma2-9B-Frags-Generic": "Gemma2-9B-Frags",
        "Gemma2-9B-Frags-Drug": "Gemma2-9B-Frags",
        "Gemma2-9B-Frags-Bio": "Gemma2-9B-Frags",
        "Gemma2-9B-Frags-Chem": "Gemma2-9B-Frags",
        "Gemma2-9B-Frags-Quant": "Gemma2-9B-Frags",
        "Gemma2-27B-Generic": "Gemma2-27B",
        "Gemma2-27B-Drug": "Gemma2-27B",
        "Gemma2-27B-Bio": "Gemma2-27B",
        "Gemma2-27B-Chem": "Gemma2-27B",
        "Gemma2-27B-Quant": "Gemma2-27B",
        "Gemma2-27B-Frags-Generic": "Gemma2-27B-Frags",
        "Gemma2-27B-Frags-Drug": "Gemma2-27B-Frags",
        "Gemma2-27B-Frags-Bio": "Gemma2-27B-Frags",
        "Gemma2-27B-Frags-Chem": "Gemma2-27B-Frags",
        "Gemma2-27B-Frags-Quant": "Gemma2-27B-Frags",
        "MistralNeMo-12B-Generic": "MistralNeMo-12B",
        "MistralNeMo-12B-Drug": "MistralNeMo-12B",
        "MistralNeMo-12B-Bio": "MistralNeMo-12B",
        "MistralNeMo-12B-Chem": "MistralNeMo-12B",
        "MistralNeMo-12B-Quant": "MistralNeMo-12B",
        "MistralNeMo-12B-Frags-Generic": "MistralNeMo-12B-Frags",
        "MistralNeMo-12B-Frags-Drug": "MistralNeMo-12B-Frags",
        "MistralNeMo-12B-Frags-Bio": "MistralNeMo-12B-Frags",
        "MistralNeMo-12B-Frags-Chem": "MistralNeMo-12B-Frags",
        "MistralNeMo-12B-Frags-Quant": "MistralNeMo-12B-Frags",
        "": "",
        "": "",
    }

# Grouping General LLMs by size
if grouping == "size":
    groups = {
        "Llama3-8B-Generic": "General LLM Small",
        "Llama3-8B-Drug": "General LLM Small",
        "Llama3-8B-Bio": "General LLM Small",
        "Llama3-8B-Chem": "General LLM Small",
        "Llama3-8B-Quant": "General LLM Small",
        "Llama3-8B-Frags-Generic": "General LLM Small",
        "Llama3-8B-Frags-Drug": "General LLM Small",
        "Llama3-8B-Frags-Bio": "General LLM Small",
        "Llama3-8B-Frags-Chem": "General LLM Small",
        "Llama3-8B-Frags-Quant": "General LLM Small",
        "Llama3.1-8B-Generic": "General LLM Small",
        "Llama3.1-8B-Drug": "General LLM Small",
        "Llama3.1-8B-Bio": "General LLM Small",
        "Llama3.1-8B-Chem": "General LLM Small",
        "Llama3.1-8B-Quant": "General LLM Small",
        "Llama3.1-8B-Frags-Generic": "General LLM Small",
        "Llama3.1-8B-Frags-Drug": "General LLM Small",
        "Llama3.1-8B-Frags-Bio": "General LLM Small",
        "Llama3.1-8B-Frags-Chem": "General LLM Small",
        "Llama3.1-8B-Frags-Quant": "General LLM Small",
        "Llama3-70B-Generic": "General LLM Large",
        "Llama3-70B-Drug": "General LLM Large",
        "Llama3-70B-Bio": "General LLM Large",
        "Llama3-70B-Chem": "General LLM Large",
        "Llama3-70B-Quant": "General LLM Large",
        "Llama3-70B-Frags-Generic": "General LLM Large",
        "Llama3-70B-Frags-Drug": "General LLM Large",
        "Llama3-70B-Frags-Bio": "General LLM Large",
        "Llama3-70B-Frags-Chem": "General LLM Large",
        "Llama3-70B-Frags-Quant": "General LLM Large",
        "Llama3.1-70B-Generic": "General LLM Large",
        "Llama3.1-70B-Drug": "General LLM Large",
        "Llama3.1-70B-Bio": "General LLM Large",
        "Llama3.1-70B-Chem": "General LLM Large",
        "Llama3.1-70B-Quant": "General LLM Large",
        "Llama3.1-70B-Frags-Generic": "General LLM Large",
        "Llama3.1-70B-Frags-Drug": "General LLM Large",
        "Llama3.1-70B-Frags-Bio": "General LLM Large",
        "Llama3.1-70B-Frags-Chem": "General LLM Large",
        "Llama3.1-70B-Frags-Quant": "General LLM Large",
        "Gemma2-9B-Generic": "General LLM Small",
        "Gemma2-9B-Drug": "General LLM Small",
        "Gemma2-9B-Bio": "General LLM Small",
        "Gemma2-9B-Chem": "General LLM Small",
        "Gemma2-9B-Quant": "General LLM Small",
        "Gemma2-9B-Frags-Generic": "General LLM Small",
        "Gemma2-9B-Frags-Drug": "General LLM Small",
        "Gemma2-9B-Frags-Bio": "General LLM Small",
        "Gemma2-9B-Frags-Chem": "General LLM Small",
        "Gemma2-9B-Frags-Quant": "General LLM Small",
        "Gemma2-27B-Generic": "General LLM Medium",
        "Gemma2-27B-Drug": "General LLM Medium",
        "Gemma2-27B-Bio": "General LLM Medium",
        "Gemma2-27B-Chem": "General LLM Medium",
        "Gemma2-27B-Quant": "General LLM Medium",
        "Gemma2-27B-Frags-Generic": "General LLM Medium",
        "Gemma2-27B-Frags-Drug": "General LLM Medium",
        "Gemma2-27B-Frags-Bio": "General LLM Medium",
        "Gemma2-27B-Frags-Chem": "General LLM Medium",
        "Gemma2-27B-Frags-Quant": "General LLM Medium",
        "MistralNeMo-12B-Generic": "General LLM Small",
        "MistralNeMo-12B-Drug": "General LLM Small",
        "MistralNeMo-12B-Bio": "General LLM Small",
        "MistralNeMo-12B-Chem": "General LLM Small",
        "MistralNeMo-12B-Quant": "General LLM Small",
        "MistralNeMo-12B-Frags-Generic": "General LLM Small",
        "MistralNeMo-12B-Frags-Drug": "General LLM Small",
        "MistralNeMo-12B-Frags-Bio": "General LLM Small",
        "MistralNeMo-12B-Frags-Chem": "General LLM Small",
        "MistralNeMo-12B-Frags-Quant": "General LLM Small",
        "": "",
        "": "",
        "GPT-4o-Generic": "General LLM Frontier",
        "GPT-4o-Frags-Generic": "General LLM Frontier",
        "Llama3.1-405B-Generic": "General LLM Frontier",
        "Llama3.1-405B-Frags-Generic": "General LLM Frontier",
    }


# Grouping Frags vs. no Frags
if grouping == "frag":
    groups = {
        "Llama3-8B-Generic": "General LLM",
        "Llama3-8B-Drug": "General LLM",
        "Llama3-8B-Bio": "General LLM",
        "Llama3-8B-Chem": "General LLM",
        "Llama3-8B-Quant": "General LLM",
        "Llama3-8B-Frags-Generic": "General LLM Frags",
        "Llama3-8B-Frags-Drug": "General LLM Frags",
        "Llama3-8B-Frags-Bio": "General LLM Frags",
        "Llama3-8B-Frags-Chem": "General LLM Frags",
        "Llama3-8B-Frags-Quant": "General LLM Frags",
        "Llama3.1-8B-Generic": "General LLM",
        "Llama3.1-8B-Drug": "General LLM",
        "Llama3.1-8B-Bio": "General LLM",
        "Llama3.1-8B-Chem": "General LLM",
        "Llama3.1-8B-Quant": "General LLM",
        "Llama3.1-8B-Frags-Generic": "General LLM Frags",
        "Llama3.1-8B-Frags-Drug": "General LLM Frags",
        "Llama3.1-8B-Frags-Bio": "General LLM Frags",
        "Llama3.1-8B-Frags-Chem": "General LLM Frags",
        "Llama3.1-8B-Frags-Quant": "General LLM Frags",
        "Llama3-70B-Generic": "General LLM",
        "Llama3-70B-Drug": "General LLM",
        "Llama3-70B-Bio": "General LLM",
        "Llama3-70B-Chem": "General LLM",
        "Llama3-70B-Quant": "General LLM",
        "Llama3-70B-Frags-Generic": "General LLM Frags",
        "Llama3-70B-Frags-Drug": "General LLM Frags",
        "Llama3-70B-Frags-Bio": "General LLM Frags",
        "Llama3-70B-Frags-Chem": "General LLM Frags",
        "Llama3-70B-Frags-Quant": "General LLM Frags",
        "Llama3.1-70B-Generic": "General LLM",
        "Llama3.1-70B-Drug": "General LLM",
        "Llama3.1-70B-Bio": "General LLM",
        "Llama3.1-70B-Chem": "General LLM",
        "Llama3.1-70B-Quant": "General LLM",
        "Llama3.1-70B-Frags-Generic": "General LLM Frags",
        "Llama3.1-70B-Frags-Drug": "General LLM Frags",
        "Llama3.1-70B-Frags-Bio": "General LLM Frags",
        "Llama3.1-70B-Frags-Chem": "General LLM Frags",
        "Llama3.1-70B-Frags-Quant": "General LLM Frags",
        "Gemma2-9B-Generic": "General LLM",
        "Gemma2-9B-Drug": "General LLM",
        "Gemma2-9B-Bio": "General LLM",
        "Gemma2-9B-Chem": "General LLM",
        "Gemma2-9B-Quant": "General LLM",
        "Gemma2-9B-Frags-Generic": "General LLM Frags",
        "Gemma2-9B-Frags-Drug": "General LLM Frags",
        "Gemma2-9B-Frags-Bio": "General LLM Frags",
        "Gemma2-9B-Frags-Chem": "General LLM Frags",
        "Gemma2-9B-Frags-Quant": "General LLM Frags",
        "Gemma2-27B-Generic": "General LLM",
        "Gemma2-27B-Drug": "General LLM",
        "Gemma2-27B-Bio": "General LLM",
        "Gemma2-27B-Chem": "General LLM",
        "Gemma2-27B-Quant": "General LLM",
        "Gemma2-27B-Frags-Generic": "General LLM Frags",
        "Gemma2-27B-Frags-Drug": "General LLM Frags",
        "Gemma2-27B-Frags-Bio": "General LLM Frags",
        "Gemma2-27B-Frags-Chem": "General LLM Frags",
        "Gemma2-27B-Frags-Quant": "General LLM Frags",
        "MistralNeMo-12B-Generic": "General LLM",
        "MistralNeMo-12B-Drug": "General LLM",
        "MistralNeMo-12B-Bio": "General LLM",
        "MistralNeMo-12B-Chem": "General LLM",
        "MistralNeMo-12B-Quant": "General LLM",
        "MistralNeMo-12B-Frags-Generic": "General LLM Frags",
        "MistralNeMo-12B-Frags-Drug": "General LLM Frags",
        "MistralNeMo-12B-Frags-Bio": "General LLM Frags",
        "MistralNeMo-12B-Frags-Chem": "General LLM Frags",
        "MistralNeMo-12B-Frags-Quant": "General LLM Frags",
        "": "",
        "": "",
        "GPT-4o-Generic": "General LLM",
        "GPT-4o-Frags-Generic": "General LLM Frags",
        "Llama3.1-405B-Generic": "General LLM",
        "Llama3.1-405B-Frags-Generic": "General LLM Frags",
    }


# Grouping across prompt variations and frags
if grouping == "prompt_var_frag":
    groups = {
        "Llama3-8B-Generic": "Llama3-8B",
        "Llama3-8B-Drug": "Llama3-8B",
        "Llama3-8B-Bio": "Llama3-8B",
        "Llama3-8B-Chem": "Llama3-8B",
        "Llama3-8B-Quant": "Llama3-8B",
        "Llama3-8B-Frags-Generic": "Llama3-8B",
        "Llama3-8B-Frags-Drug": "Llama3-8B",
        "Llama3-8B-Frags-Bio": "Llama3-8B",
        "Llama3-8B-Frags-Chem": "Llama3-8B",
        "Llama3-8B-Frags-Quant": "Llama3-8B",
        "Llama3.1-8B-Generic": "Llama3.1-8B",
        "Llama3.1-8B-Drug": "Llama3.1-8B",
        "Llama3.1-8B-Bio": "Llama3.1-8B",
        "Llama3.1-8B-Chem": "Llama3.1-8B",
        "Llama3.1-8B-Quant": "Llama3.1-8B",
        "Llama3.1-8B-Frags-Generic": "Llama3.1-8B",
        "Llama3.1-8B-Frags-Drug": "Llama3.1-8B",
        "Llama3.1-8B-Frags-Bio": "Llama3.1-8B",
        "Llama3.1-8B-Frags-Chem": "Llama3.1-8B",
        "Llama3.1-8B-Frags-Quant": "Llama3.1-8B",
        "Llama3-70B-Generic": "Llama3-70B",
        "Llama3-70B-Drug": "Llama3-70B",
        "Llama3-70B-Bio": "Llama3-70B",
        "Llama3-70B-Chem": "Llama3-70B",
        "Llama3-70B-Quant": "Llama3-70B",
        "Llama3-70B-Frags-Generic": "Llama3-70B",
        "Llama3-70B-Frags-Drug": "Llama3-70B",
        "Llama3-70B-Frags-Bio": "Llama3-70B",
        "Llama3-70B-Frags-Chem": "Llama3-70B",
        "Llama3-70B-Frags-Quant": "Llama3-70B",
        "Llama3.1-70B-Generic": "Llama3.1-70B",
        "Llama3.1-70B-Drug": "Llama3.1-70B",
        "Llama3.1-70B-Bio": "Llama3.1-70B",
        "Llama3.1-70B-Chem": "Llama3.1-70B",
        "Llama3.1-70B-Quant": "Llama3.1-70B",
        "Llama3.1-70B-Frags-Generic": "Llama3.1-70B",
        "Llama3.1-70B-Frags-Drug": "Llama3.1-70B",
        "Llama3.1-70B-Frags-Bio": "Llama3.1-70B",
        "Llama3.1-70B-Frags-Chem": "Llama3.1-70B",
        "Llama3.1-70B-Frags-Quant": "Llama3.1-70B",
        "Gemma2-9B-Generic": "Gemma2-9B",
        "Gemma2-9B-Drug": "Gemma2-9B",
        "Gemma2-9B-Bio": "Gemma2-9B",
        "Gemma2-9B-Chem": "Gemma2-9B",
        "Gemma2-9B-Quant": "Gemma2-9B",
        "Gemma2-9B-Frags-Generic": "Gemma2-9B",
        "Gemma2-9B-Frags-Drug": "Gemma2-9B",
        "Gemma2-9B-Frags-Bio": "Gemma2-9B",
        "Gemma2-9B-Frags-Chem": "Gemma2-9B",
        "Gemma2-9B-Frags-Quant": "Gemma2-9B",
        "Gemma2-27B-Generic": "Gemma2-27B",
        "Gemma2-27B-Drug": "Gemma2-27B",
        "Gemma2-27B-Bio": "Gemma2-27B",
        "Gemma2-27B-Chem": "Gemma2-27B",
        "Gemma2-27B-Quant": "Gemma2-27B",
        "Gemma2-27B-Frags-Generic": "Gemma2-27B",
        "Gemma2-27B-Frags-Drug": "Gemma2-27B",
        "Gemma2-27B-Frags-Bio": "Gemma2-27B",
        "Gemma2-27B-Frags-Chem": "Gemma2-27B",
        "Gemma2-27B-Frags-Quant": "Gemma2-27B",
        "MistralNeMo-12B-Generic": "MistralNeMo-12B",
        "MistralNeMo-12B-Drug": "MistralNeMo-12B",
        "MistralNeMo-12B-Bio": "MistralNeMo-12B",
        "MistralNeMo-12B-Chem": "MistralNeMo-12B",
        "MistralNeMo-12B-Quant": "MistralNeMo-12B",
        "MistralNeMo-12B-Frags-Generic": "MistralNeMo-12B",
        "MistralNeMo-12B-Frags-Drug": "MistralNeMo-12B",
        "MistralNeMo-12B-Frags-Bio": "MistralNeMo-12B",
        "MistralNeMo-12B-Frags-Chem": "MistralNeMo-12B",
        "MistralNeMo-12B-Frags-Quant": "MistralNeMo-12B",
        "GPT-4o-Frags-Generic": "GPT-4o",
        "GPT-4o-Generic": "GPT-4o",
        "Llama3.1-405B-Generic": "Llama3.1-405B",
        "Llama3.1-405B-Frags-Generic": "Llama3.1-405B",
    }


# Grouping into prompt variation
if grouping == "prompt_var":
    groups = {
        "Llama3-8B-Generic": "Generic Prompt",
        "Llama3-8B-Drug": "Drug Prompt",
        "Llama3-8B-Bio": "Biology Prompt",
        "Llama3-8B-Chem": "Chemistry Prompt",
        "Llama3-8B-Quant": "Quantum Prompt",
        "Llama3-8B-Frags-Generic": "Generic Prompt",
        "Llama3-8B-Frags-Drug": "Drug Prompt",
        "Llama3-8B-Frags-Bio": "Biology Prompt",
        "Llama3-8B-Frags-Chem": "Chemistry Prompt",
        "Llama3-8B-Frags-Quant": "Quantum Prompt",
        "Llama3.1-8B-Generic": "Generic Prompt",
        "Llama3.1-8B-Drug": "Drug Prompt",
        "Llama3.1-8B-Bio": "Biology Prompt",
        "Llama3.1-8B-Chem": "Chemistry Prompt",
        "Llama3.1-8B-Quant": "Quantum Prompt",
        "Llama3.1-8B-Frags-Generic": "Generic Prompt",
        "Llama3.1-8B-Frags-Drug": "Drug Prompt",
        "Llama3.1-8B-Frags-Bio": "Biology Prompt",
        "Llama3.1-8B-Frags-Chem": "Chemistry Prompt",
        "Llama3.1-8B-Frags-Quant": "Quantum Prompt",
        "Llama3-70B-Generic": "Generic Prompt",
        "Llama3-70B-Drug": "Drug Prompt",
        "Llama3-70B-Bio": "Biology Prompt",
        "Llama3-70B-Chem": "Chemistry Prompt",
        "Llama3-70B-Quant": "Quantum Prompt",
        "Llama3-70B-Frags-Generic": "Generic Prompt",
        "Llama3-70B-Frags-Drug": "Drug Prompt",
        "Llama3-70B-Frags-Bio": "Biology Prompt",
        "Llama3-70B-Frags-Chem": "Chemistry Prompt",
        "Llama3-70B-Frags-Quant": "Quantum Prompt",
        "Llama3.1-70B-Generic": "Generic Prompt",
        "Llama3.1-70B-Drug": "Drug Prompt",
        "Llama3.1-70B-Bio": "Biology Prompt",
        "Llama3.1-70B-Chem": "Chemistry Prompt",
        "Llama3.1-70B-Quant": "Quantum Prompt",
        "Llama3.1-70B-Frags-Generic": "Generic Prompt",
        "Llama3.1-70B-Frags-Drug": "Drug Prompt",
        "Llama3.1-70B-Frags-Bio": "Biology Prompt",
        "Llama3.1-70B-Frags-Chem": "Chemistry Prompt",
        "Llama3.1-70B-Frags-Quant": "Quantum Prompt",
        "Gemma2-9B-Generic": "Generic Prompt",
        "Gemma2-9B-Drug": "Drug Prompt",
        "Gemma2-9B-Bio": "Biology Prompt",
        "Gemma2-9B-Chem": "Chemistry Prompt",
        "Gemma2-9B-Quant": "Quantum Prompt",
        "Gemma2-9B-Frags-Generic": "Generic Prompt",
        "Gemma2-9B-Frags-Drug": "Drug Prompt",
        "Gemma2-9B-Frags-Bio": "Biology Prompt",
        "Gemma2-9B-Frags-Chem": "Chemistry Prompt",
        "Gemma2-9B-Frags-Quant": "Quantum Prompt",
        "Gemma2-27B-Generic": "Generic Prompt",
        "Gemma2-27B-Drug": "Drug Prompt",
        "Gemma2-27B-Bio": "Biology Prompt",
        "Gemma2-27B-Chem": "Chemistry Prompt",
        "Gemma2-27B-Quant": "Quantum Prompt",
        "Gemma2-27B-Frags-Generic": "Generic Prompt",
        "Gemma2-27B-Frags-Drug": "Drug Prompt",
        "Gemma2-27B-Frags-Bio": "Biology Prompt",
        "Gemma2-27B-Frags-Chem": "Chemistry Prompt",
        "Gemma2-27B-Frags-Quant": "Quantum Prompt",
        "MistralNeMo-12B-Generic": "Generic Prompt",
        "MistralNeMo-12B-Drug": "Drug Prompt",
        "MistralNeMo-12B-Bio": "Biology Prompt",
        "MistralNeMo-12B-Chem": "Chemistry Prompt",
        "MistralNeMo-12B-Quant": "Quantum Prompt",
        "MistralNeMo-12B-Frags-Generic": "Generic Prompt",
        "MistralNeMo-12B-Frags-Drug": "Drug Prompt",
        "MistralNeMo-12B-Frags-Bio": "Biology Prompt",
        "MistralNeMo-12B-Frags-Chem": "Chemistry Prompt",
        "MistralNeMo-12B-Frags-Quant": "Quantum Prompt",
        "": "",
        "": "",
        "GPT-4o-Frags-Generic": "GPT-4o-Generic",
        "GPT-4o-Generic": "GPT-4o-Generic",
        "Llama3.1-405B-Generic": "Llama3.1-405B-Generic",
        "Llama3.1-405B-Frags-Generic": "Llama3.1-405B-Generic",
    }


def to_group(model):
    return groups[model] if model in groups else model


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

import glob

cap_files = glob.glob("../captions/*.csv")
caption_sources = [cf.split("/")[-1].split(".csv")[0] for cf in cap_files] + ["GNN"]
caption_sources.remove("BBBP_captioner")


for dataset in datasets:
    task = dataset_to_task[dataset]

    for cs in caption_sources:

        for i in splits:

            df = pd.read_csv(f"{data_folder}/{method}/{dataset}/{i}/{cs}.csv")

            num_features = len(
                [col for col in df.columns if col.startswith("Activity")]
            )

            for j in range(num_features):

                act_dict = {}
                error_dict = defaultdict(list)

                avg_error = defaultdict(list)

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

                for row in df.iterrows():
                    d = row[1]
                    sm = to_group(d["Source Model"])
                    if num_features > 1:
                        error_dict[sm].append(
                            (
                                d[f"Prediction_{j}"],
                                d[f"Activity_{j}"],
                                "",
                                sm,
                                d[f"Error_{j}"],
                            )
                        )
                    else:
                        error_dict[sm].append(
                            (d["Prediction"], d["Activity"], "", sm, d["Error"])
                        )

                    source_models.add(sm)

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

models = sorted(list(set([to_group(cs) for cs in caption_sources])))

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


pd.set_option("display.precision", 3)

if True:  # get correlation with ELO
    from create_ELO_from_BT import get_ratings
    from get_ELO_head2head import get_battles_h2h

    battles = get_battles_h2h(data_folder, "SVM", datasets, splits)

    battles["model_a"] = battles["model_a"].map(to_group)
    battles["model_b"] = battles["model_b"].map(to_group)

    bars = get_ratings(battles)

    bars = bars._append({"model": "GNN", "rating": np.nan}, ignore_index=True)

    bars["ROC-AUC"] = bars["model"].map(AUCs)
    bars["Average Precision"] = bars["model"].map(APs)
    bars["Loss"] = bars["model"].map(Ls)
    bars["Avg. Error"] = bars["model"].map(AEs)
    bars["ROC-AUC"] = bars["ROC-AUC"] * 100
    bars["Average Precision"] = bars["Average Precision"] * 100

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
        with open(f"latex_files/groups/{grouping}_{ds}_Pearson.txt", "w") as text_file:
            tmp = bars_na.corr(numeric_only=True).to_latex(
                index=True, float_format="{:.3f}".format
            )
            text_file.write(tmp)
        bars_na.corr(numeric_only=True).to_csv(
            f"latex_files/groups/{grouping}_{ds}_Pearson.csv", index=True
        )
        with open(f"latex_files/groups/{grouping}_{ds}_Spearman.txt", "w") as text_file:
            tmp = bars_na.corr(numeric_only=True, method="spearman").to_latex(
                index=True, float_format="{:.3f}".format
            )
            text_file.write(tmp)
        bars_na.corr(numeric_only=True, method="spearman").to_csv(
            f"latex_files/groups/{grouping}_{ds}_Pearson.csv", index=True
        )
        with open(f"latex_files/groups/{grouping}_{ds}_table.txt", "w") as text_file:
            tmp = bars.to_latex(index=True, float_format="{:.3f}".format)
            text_file.write(tmp)
        bars.to_csv(f"latex_files/groups/{grouping}_{ds}_table.csv", index=True)
