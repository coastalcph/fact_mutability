import os
import json
from collections import defaultdict, Counter
from datasets import load_dataset
from wikidata.client import Client
from tqdm import tqdm
from multiprocessing import Pool, Manager

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


def main():
    client = Client()
    train = build_dataset('data/templama/train_with_aliases.json')
    val = build_dataset('data/templama/val_with_aliases.json')
    for query in val:
        train.add_query(query)
    test = build_dataset('data/templama/test_with_aliases.json')
    for query in test:
        train.add_query(query)
    stats = defaultdict(int)

    if not os.path.exists('./data/qids_to_label.json'):
        subjs = {q.id.split("_")[0] for q in train}
        with Pool(24) as p:
            results = tqdm(p.imap(fetch_qids_text, subjs), total=len(subjs))
            labels = list(results)
        qids_to_label = {q: l for q, l in labels}
        json.dump(qids_to_label, open('./data/qids_to_label.json', 'w'))
    else:
        qids_to_label = json.load(open('./data/qids_to_label.json'))


    payloads = list()
    for query in tqdm(train):
        subj, relation = query.id.split("_") 
        entity_text = qids_to_label[subj]
        answers = query.answers
        for answer in answers:
            for text in answer.texts:
                payloads.append((entity_text, text.lower(), subj, answer.qcode))
                # subj_hit, obj_hit, cooccurrence_hit = find_cooccurrences(inputs, entity_text, text.lower())
    print("Payloads", len(payloads))
    for p in payloads[:10]:
        print(p)
        

    # dataset = load_dataset("conceptofmind/flan2021_submix_original")
    # inputs = format_dataset(dataset['train'])
    # with open("./data/flan_training.txt", "w") as fhandle:
    #     for i in tqdm(inputs):
    #         i = i.replace('\n', ' ')
    #         fhandle.write(f"{i}\n")

    # for input in tqdm(dataset['train']['inputs']):
    #     input = input.lower()
    #     res = task(input, payloads)
    #     print(res)


    # for query in tqdm(train):
    #     subj, relation = query.id.split("_") 
    #     entity = client.get(subj, load=True)
    #     entity_text = str(entity.label).lower()
    #     qids_to_label[subj] = entity_text


    for query in tqdm(train):
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
    json.dump(stats, open("./data/templama_cooccurrences.json", "w"), indent=True)

if __name__ == '__main__':
    main()
