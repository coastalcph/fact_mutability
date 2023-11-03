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
    for type, rels in relations_by_type.items():
        f1s = list()
        confidences = list()
        for rel in rels:
            template = alpaca_templates[rel]
            stats = pd.read_json(f"./output/alpaca_{template}/{rel}_results_per_example.json")
            predictions = load_predictions(f"../fm_predictions/fm_queries_v2/alpaca_fmv2_final_{template}---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json")  # LAMA
            for i, row in stats.iterrows():
                query_id = row['id']
                f1 = row['f1']
                confidence = predictions[query_id]['predictions'][0]['first_token_probability']
                f1s.append(f1)
                confidences.append(confidence)
        f1s = np.array(f1s)
        confidences = np.array(confidences)
        bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        digitized = np.digitize(f1s, bins, right=True)
        bin_means = [confidences[digitized == i].mean() for i in range(0, len(bins))]
        bin_std = [confidences[digitized == i].std() for i in range(0, len(bins))]
        bin_counts = [len(confidences[digitized == i]) for i in range(0, len(bins))]
        print(type)
        for m, s, c in zip(bin_means, bin_std, bin_counts):
            print(m, s, c)
        # print(bin_means)
        # print(bin_std)
        # print(bin_counts)
            
    

if __name__ == '__main__':
    main()
