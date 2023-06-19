from collections import defaultdict
import json

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from mutability.domain import Relation
from utils.data_handling import *

np.random.seed(42)

def build_relations(dataset):
    relations_mutations = defaultdict(list)
    for query in dataset:
        relations_mutations[query.relation_id].append(query.get_ratio())
    relations = dict()
    for relation, mutations in relations_mutations.items():
        mean = np.mean(mutations)
        std = np.std(mutations)
        relation_obj = Relation(relation, mean, std) 
        relations[relation] = relation_obj
    return relations


def main():
    for split in ['val']:
        immutable_dataset = build_dataset('data/immutable_with_aliases.json')
        dataset = build_dataset('data/{}_with_aliases.json'.format(split))
        predictions_imm = load_predictions('data/predictions-imm-alpaca.json')
        predictions_mu = load_predictions('data/predictions-mut-alpaca.json')

        relations = build_relations(dataset)
        for relation_id, relation in relations.items():
            print(relation_id, relation.mutation_mean, relation.mutation_std)

        # compute changes
        ratios = list()
        mutables = {
            "imm": defaultdict(list),
            "never": defaultdict(list),
            "rarely": defaultdict(list),
            "often": defaultdict(list)
        }

        # Fetch mutation rate (0.0) and confidence for immutable dataset
        for query in immutable_dataset:
            prediction = get_prediction(predictions_imm, query.id)
            confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
            confidences = [c for c in confidences if not np.isnan(c)]
            confidence = confidences[0]
            ratio = 0.0
            mutables['imm']['ratios'].append(ratio)
            mutables['imm']['confidences'].append(confidence)
            mutables['imm']['average'].append(np.mean(confidences))

        for query in dataset:
            ratio = query.get_ratio()
            relation = relations[query.relation_id]
            relation_ratio = relation.sample_mutation_rate()
            ratio += 0.5 * relation_ratio
            prediction = get_prediction(predictions_mu, query.id)
            confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
            confidences = [c for c in confidences if not np.isnan(c)]
            confidence = confidences[0]
            if ratio < 0.3:
                mutables['never']['ratios'].append(ratio)
                mutables['never']['confidences'].append(confidence)
                mutables['never']['average'].append(np.mean(confidences))
            elif ratio < 0.6:
                mutables['rarely']['ratios'].append(ratio)
                mutables['rarely']['confidences'].append(confidence)
                mutables['rarely']['average'].append(np.mean(confidences))
            elif ratio >= 0.6:
                mutables['often']['ratios'].append(ratio)
                mutables['often']['confidences'].append(confidence)
                mutables['often']['average'].append(np.mean(confidences))
        
        all_confidences = list()
        all_ratios = list()
        for mutation, data in mutables.items():
            print(mutation)
            ratios = data['ratios']
            confidences = data['confidences']
            averages = data['average']
            all_confidences += confidences
            all_ratios += ratios 
            print("Data points", len(averages))
            print("Average of averages (std)", np.mean(averages), np.std(averages))
            print("Average (std)", np.mean(confidences), np.std(confidences))
            print("Max", np.max(confidences))
            print("Min", np.min(confidences))
        corr = pearsonr(all_confidences, all_ratios)
        print("Correlation", corr)
        plt.scatter(all_ratios, all_confidences)
        plt.ylabel("Confidence")
        plt.xlabel("Mutation rate")
        plt.show()


if __name__ == '__main__':
    main()
