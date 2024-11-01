import pandas as pd

from tqdm import tqdm

tqdm.pandas()

import math

import pandas as pd

from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose

from vllm import LLM, SamplingParams


def create_captions(
    df,
    batch_size,
    sys_prompt="You are a helpful expert in medicinal chemistry. You are sharing your knowledge of known chemistry with a colleague.",
    prompt="The following molecule might be a drug. Tell me properties of the molecule, such as the mechanism of action, class, and target, which might be relevant.",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
):

    name = model_name.split("/")[-1]

    llm = LLM(model=model_name, gpu_memory_utilization=0.8, max_model_len=1024)

    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=512, min_p=0.15, top_p=0.85
    )

    tokenizer = llm.get_tokenizer()

    def get_history(smi):

        history = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": prompt + "\nMolecular Fragments: {}\n".format(smi),
            },
        ]
        return history

    bad_mols = set()

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

    # df['BRICS'] = df['SMILES'].map(create_BRICS) #already calculated

    def divide_chunks(l, n):

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def get_captions(frags):
        all_caps = []

        # for fbatch in tqdm(divide_chunks(frags, batch_size), total=math.ceil(len(frags)/batch_size)):
        prompts = tokenizer.apply_chat_template(
            [get_history(str(list(f))) for f in frags],
            tokenize=False,
        )

        caps = llm.generate(prompts, sampling_params)
        caps = [c.outputs[0].text for c in caps]
        caps = [
            c.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
            for c in caps
        ]

        all_caps.extend(caps)

        return all_caps

    captions = get_captions(df["BRICS"].tolist())

    df["captions"] = captions

    return df, name
