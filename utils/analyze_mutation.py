from collections import defaultdict
import json

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from data_handling import *

def main():
    for split in ['val']:
        dataset = build_dataset('data/{}_with_aliases.json'.format(split))
        predictions = load_predictions('data/predictions.json')

        # compute changes
        ratios = list()
        mutables = {
            "never": defaultdict(list),
            "rarely": defaultdict(list),
            "often": defaultdict(list)
        }
        for query in dataset:
            ratio = query.get_ratio()
            prediction = get_prediction(predictions, query.id)
            confidences = sorted([p['token_scores'][1] if p['answer'].lower().startswith("the ") else p['token_scores'][0]  for p in prediction['predictions'] if len(p['token_scores'])], reverse=True)
            confidences = [c for c in confidences if not np.isnan(c)]
            confidence = confidences[0]
            if ratio < 0.2:
                mutables['never']['ratios'].append(ratio)
                mutables['never']['confidences'].append(confidence)
                mutables['never']['average'].append(np.mean(confidences))
            elif ratio < 0.5:
                mutables['rarely']['ratios'].append(ratio)
                mutables['rarely']['confidences'].append(confidence)
                mutables['rarely']['average'].append(np.mean(confidences))
            elif ratio >= 0.5:
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
