import argparse
import os
from collections import Counter
from functools import partial

import numpy as np
import torch
import json
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
from glob import glob
from inference import prepare_prompt, DEF_TEMPLATE_TO_USE, DEF_INSTRUCTION


def replace_subject(prompt_format, tokenizer, prepare_prompt_func, example):
    query = example["query"].replace("_X_ .", "_X_.")
    text = query.replace("_X_.", example["answer"][0]["name"]).strip()
    if prompt_format != "{}":
        text = text[0].lower() + text[1:]
    text = prompt_format.format(text)
    text = prepare_prompt_func(text).strip()
    return {"text": text, **tokenizer(text)}


def get_ds(relations, tokenizer, model_name, clf_mutability="immutable_1"):
    ds = load_dataset("coastalcph/fm_queries")
    ds["all_fm"] = ds["train"]
    ds.pop("train")
    for relation in relations:
        rng = np.random.default_rng(int(relation[1:]))
        ds[relation] = ds["all_fm"].filter(lambda ex: ex["relation"] == relation)
        # Select one of the 5 templates at random.
        tuple_id_to_index = {
            s: i
            for i, s in enumerate(list(set([id_[:-2] for id_ in ds[relation]["id"]])))
        }
        assert int(len(ds[relation]) / 5) == len(tuple_id_to_index)
        template_choice = rng.choice(5, int(len(ds[relation]) / 5))
        ds[relation] = ds[relation].filter(
            lambda ex: template_choice[tuple_id_to_index[ex["id"][:-2]]]
            == int(ex["id"][-1])
        )
    ds.pop("all_fm")
    ds = ds.filter(lambda ex: len(ex["answer"]) > 0)
    ds = ds.map(lambda ex: {"label": 1 if ex["type"] == "mutable" else 0})
    if clf_mutability == "immutable_1":
        ds = ds.filter(lambda ex: ex["type"] != "immutable_n")
    elif clf_mutability == "immutable_n":
        ds = ds.filter(lambda ex: ex["type"] != "immutable")
    return ds.map(
        partial(
            replace_subject,
            "{}",
            tokenizer,
            lambda q: prepare_prompt(
                q, model_name, DEF_INSTRUCTION, DEF_TEMPLATE_TO_USE
            ),
        )
    )


def preprocess_ds_by_freq(ds, frequency_files_pattern):
    relation_to_count_filename = {}
    for f in glob(frequency_files_pattern):
        relation = os.path.basename(f)[: -len("_with_counts.json")]
        if relation in relation_to_count_filename:
            if "yellow" in relation_to_count_filename[relation]:
                relation_to_count_filename[relation] = f
            continue
        relation_to_count_filename[relation] = f

    all_counts = []
    for relation in relations:
        with open(relation_to_count_filename[relation]) as f:
            subj_count = json.load(f)[:1500]
        subj_count = {s: c for s, c in subj_count}
        all_counts.extend(subj_count.values())
        ds[relation] = ds[relation].map(
            lambda ex: {"subj_count": subj_count[ex["id"].split("_")[0]]}
        )
    ds = concatenate_datasets([ds[r] for r in relations])
    freq_splits = {}
    percentiles = np.percentile(all_counts, np.arange(10, 101, 10))
    lower_bound = 0
    for i, percentile in enumerate(percentiles):
        freq_splits[f"percentile_{i}"] = ds.filter(
            lambda ex: ex["subj_count"] > lower_bound and ex["subj_count"] <= percentile
        )
        lower_bound = percentile
        print(f"percentile_{i}", len(freq_splits[f"percentile_{i}"]))
    return DatasetDict(freq_splits)


model_name = "/projects/nlp/data/constanzam/llama/huggingface-ckpts/7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
frequency_files_pattern = "dataset/data/*wikidata_objs_freq/*_with_counts.json"

relations = [
    "P530",
    "P47",
    "P1303",
    "P166",
    "P264",
    "P451",
    "P210",
    "P54",
    "P551",
    "P1308",
    "P136",
    "P1037",
    "P39",
]
tokenized_ds = get_ds(relations, tokenizer, model_name, clf_mutability="immutable_n")
tokenized_ds = preprocess_ds_by_freq(tokenized_ds, frequency_files_pattern)
for i in range(10):
    print(i, Counter(tokenized_ds[f"percentile_{i}"]["type"]))

relations = [
    "P495",
    "P740",
    "P36",
    "P449",
    "P264",
    "P451",
    "P30",
    "P138",
    "P210",
    "P54",
    "P551",
    "P1308",
    "P1037",
    "P39",
]
tokenized_ds = get_ds(relations, tokenizer, model_name, clf_mutability="immutable_1")
tokenized_ds = preprocess_ds_by_freq(tokenized_ds, frequency_files_pattern)
for i in range(10):
    print(i, Counter(tokenized_ds[f"percentile_{i}"]["type"]))
