from datasets import load_dataset
from glob import glob
import json
import os
import pandas as pd
import numpy as np

frequency_files_pattern = "dataset/data/*wikidata_objs_freq/*_with_counts.json"
ds = load_dataset("coastalcph/fm_queries")["train"]
relation_to_count_filename = {}
for f in glob(frequency_files_pattern):
    relation = os.path.basename(f)[: -len("_with_counts.json")]
    if relation in relation_to_count_filename:
        if "yellow" in relation_to_count_filename[relation]:
            relation_to_count_filename[relation] = f
        continue
    relation_to_count_filename[relation] = f
subj_count_all = {}
for relation in relation_to_count_filename.keys():
    with open(relation_to_count_filename[relation]) as f:
        subj_count = json.load(f)[:1500]
    for s, c in subj_count:
        if s in subj_count_all and np.abs(subj_count_all[s] - c) > 2:
            print("mismatch:", s, c, subj_count_all[s])
            subj_count_all[s] = max(c, subj_count_all[s])
        else:
            subj_count_all[s] = c
ds = ds.map(lambda ex: {"subj_count": subj_count_all[ex["id"].split("_")[0]]})

df = pd.DataFrame({"id": ds["id"], "type": ds["type"], "subj_count": ds["subj_count"]})
df["subj_obj"] = df.apply(lambda ex: ex["id"][:-1], axis=1)
df = df.drop(columns=["id"]).drop_duplicates()
df[["type", "subj_count"]].groupby(by=["type"]).agg([len, np.mean, np.std])
