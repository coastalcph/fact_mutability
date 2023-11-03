import pandas as pd
import os
from collections import defaultdict
import json

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from python2latex import Document, Table, bold

from mutability.domain import Relation
from utils.data_handling import *


from dataset import relations, relations_by_type, sorted_relations_by_type


np.random.seed(42)

def find_query_by_subject(queries, subj):
    for q in queries:
        if q['qid'] == subj:
            return q

def main():
    rels_labels = list()
    for type, rels in sorted_relations_by_type.items():
        for rel in rels:
            rels_labels.append(rel)
        rels_labels.append("Avg")
    rels_labels.append("Average")

    rels = [r for rels in sorted_relations_by_type.values() for r in rels]
    stats_by_model = defaultdict(lambda: defaultdict(float))
    template_by_model = defaultdict(lambda: defaultdict(int))
    models = ["alpaca", "llama"]
    templates = [0, 1, 2, 3, 4]

    for m in models:
        f1_by_rel = defaultdict(list)
        for t in templates:
            with open(f"./output/{m}_{t}/metrics.jsonl") as fhandle:
                for line in fhandle:
                    rel_metric = json.loads(line)
                    for rel, metric in rel_metric.items():
                        f1_by_rel[rel].append(metric)

        for rel, f1s in f1_by_rel.items():
            if rel != "all":
                stats_by_model[m][rel] = max(f1s)
                template = np.argmax(f1s)
                template_by_model[m][rel] = template

    alpaca_templates = template_by_model['alpaca']
    for type, rels in sorted_relations_by_type.items():
        print(type)
        queries_for_type = list()
        for rel in rels:
            print(rel)
            template = alpaca_templates[rel]
            selected_queries = list()
            subjects = json.load(open("./data/wikidata/objects_by_freq/{}.json".format(rel)))
            stats = pd.read_json(f"./output/alpaca_{template}/{rel}_results_per_example.json")
            predictions = load_predictions(f"../fm_predictions/fm_queries_v2/alpaca_fmv2_final_{template}---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json")  # LAMA
            good_predictions = stats[stats['f1']==1.0]
            for query_id in good_predictions['id'].tolist():
                pred = predictions[query_id]
                subj = query_id.split("_")[0]
                query = find_query_by_subject(subjects, subj)
                example = {
                    "query": query,
                    "prediction": pred
                }
                selected_queries.append(example)
            confidences = np.array([s['prediction']['predictions'][0]['first_token_probability'] for s in selected_queries])
            bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            digitized = np.digitize(confidences, bins, right=True)
            bin_counts = [len(confidences[digitized == i]) for i in range(0, len(bins))]
            for b, c in zip(bins, bin_counts):
                print(b, ";", c)
            queries_for_type += selected_queries
            q_high_conf = [s for s in selected_queries if s['prediction']['predictions'][0]['first_token_probability'] >= 0.9]
            json.dump(q_high_conf, open(f"./data/selected_examples/{rel}.json", "w"), indent=True)

            ### Analysis of selected examples
            print("Num examples", len(selected_queries))
        
        confidences = list()
        for q in queries_for_type:
            confidence = q['prediction']['predictions'][0]['first_token_probability']
            confidences.append(confidence)
        print(f"Average confidence for {type}", np.mean(confidences))
    

if __name__ == '__main__':
    main()
