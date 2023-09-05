import json
from collections import defaultdict
from datasets import load_dataset
from wikidata.client import Client
from tqdm import tqdm

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
        if subj in i and obj in i:
            cooccurrence_hit += 1
    return subj_hit, obj_hit, cooccurrence_hit


def format_dataset(dataset):
    inputs = set()
    for example in tqdm(dataset['inputs']):
        inputs.add(example.lower())
    return inputs

def main():
    client = Client()
    immutable_dataset = build_dataset('data/val_with_aliases.json')  # LAMA
    dataset = load_dataset("conceptofmind/flan2021_submix_original")
    inputs = format_dataset(dataset['train'])
    stats = defaultdict(int)
    qids_to_label = dict()

    for query in tqdm(immutable_dataset):
        subj, relation = query.id.split("_") 
        entity = client.get(subj, load=True)
        entity_text = str(entity.label).lower()
        qids_to_label[subj] = entity_text

    for query in tqdm(immutable_dataset):
        subj, relation = query.id.split("_") 
        entity_text = qids_to_label[subj]
        answers = query.answers
        for answer in tqdm(answers):
            for text in answer.texts:
                subj_hit, obj_hit, cooccurrence_hit = find_cooccurrences(inputs, entity_text, text.lower())
                # if subj_hit > 0:
                #     stats[subj] += subj_hit
                # if obj_hit > 0:
                #     stats[text] += obj_hit
                if cooccurrence_hit > 0:
                    stats[f"{subj}-{answer.qcode}"] += cooccurrence_hit
    json.dump(stats, open("./data/cooccurrences.json", "w"), indent=True)

if __name__ == '__main__':
    main()
