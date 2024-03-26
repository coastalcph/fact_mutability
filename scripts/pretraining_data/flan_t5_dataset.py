import os
import json
from collections import defaultdict, Counter
from datasets import load_dataset
from wikidata.client import Client
from tqdm import tqdm
from multiprocessing import Pool, Manager
from dataset import relations
import click
import numpy as np

from utils.data_handling import *

def find_cooccurrences(inputs, subj, obj):
    subj_hit = 0
    obj_hit = 0
    cooccurrence_hit = 0
    for i in inputs:
        # if subj in i:
        #     subj_hit += 1
        # if obj in i:
        #     obj_hit += 1
        if f" {subj} " in i and f" {obj} " in i:
            cooccurrence_hit += 1
    return subj_hit, obj_hit, cooccurrence_hit


def format_dataset(dataset):
    inputs = set()
    for example in tqdm(dataset['inputs']):
        inputs.add(example.lower())
    return inputs


def fetch_qids_text(subj):
    client = Client()
    entity = client.get(subj, load=True)
    entity_text = str(entity.label).lower()
    return subj, entity_text


def _task(pay, i):
    subj, obj = pay
    if subj in i and obj in i:
        Counter({f"{subj}-{obj}": 1})
    else:
        Counter({f"{subj}-{obj}": 0})


def task(input, payloads):
    d = defaultdict()
    with Pool(1) as p:
        res = p.imap(sum, p.starmap(_task, [(pay, input) for pay in payloads]))
    return res


def load_queries(data_path):
    unique_queries = dict()
    queries = load_dataset(data_path, split="train")
    for query in queries:
        query_id = "_".join(query['id'].split("_")[:2])
        if query_id not in unique_queries and len(query['answer']):
            unique_queries[query_id] = query
    return unique_queries


def get_chunk(dataset, chunk):
    num_examples = len(dataset)

@click.argument("chunk")
@click.command()
def main(chunk):
    chunk = int(chunk)
    print("Running on chunk", chunk)

    print("Loading pairs")
    pairs = defaultdict(set)
    for relation, cls in relations.items():
        subjects = json.load(open("./data/wikidata/objects_by_freq/{}.json".format(relation)))
        for subject in subjects:
            subj_label = subject['label']
            for a in subject['objects']:
                pairs[relation].add((subj_label.lower(), a['label'].lower()))

    print("Loading dataset")
    dataset = load_dataset("DataProvenanceInitiative/flan2021_submix_original")
    chunksize = int(len(dataset['train']) / 100)
    print("Chunksize", chunksize)
    from_chunk = chunk * chunksize
    to_chunk = (chunk+1) * chunksize
    print("From, to", from_chunk, to_chunk)
    dataset_chunk = dataset['train'][from_chunk:to_chunk]
    print(f"Formatting chunk {chunk}")
    inputs = format_dataset(dataset_chunk)

    print("Computing co-occurrences")
    stats = defaultdict(int)
    for relation, ps in tqdm(pairs.items()):
        for subj, obj in tqdm(ps):
            subj_hit, obj_hit, cooccurrence_hit = find_cooccurrences(inputs, subj, obj)
            if cooccurrence_hit > 0:
                stats[f"{subj}|{obj}"] += cooccurrence_hit

    json.dump(stats, open(f"./data/flant5_cooccurrences_{chunk}.json", "w"), indent=True)

if __name__ == '__main__':
    main()
