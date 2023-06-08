from collections import defaultdict
import json

import numpy as np
from scipy.stats import pearsonr

from mutability.domain import Queries, Query, Answer


def build_dataset(data_path):
    queries = dict()
    queries_obj = Queries()
    for line in open(data_path):
        data = json.loads(line)
        query_id = "_".join(data['id'].split("_")[:2])
        query = data['query']
        year = data['date']
        if query_id not in queries:
            queries[query_id] = {
                "query": query,
                "answers": list()
            }
        for answer in data['answer']:
            queries[query_id]['answers'].append((answer['name'], year))
    
    for query_id, data in queries.items():
        query = data['query']
        answers = data['answers']
        answers_obj = [Answer(a, y) for a, y in answers]
        query_obj = Query(query_id, query, answers_obj)
        queries_obj.add_query(query_obj)
    
    return queries_obj


def load_predictions():
    predictions = list()
    with open('./data/predictions.json') as fhandle:
        for line in fhandle:
            data = json.loads(line)
            predictions.append(data)
    return predictions

def get_prediction(predictions, qcode):
    for prediction in predictions:
        if qcode == prediction['qcode']:
            return prediction

def main():
    for split in ['val']:
        dataset = build_dataset('data/{}_with_aliases.json'.format(split))
        predictions = load_predictions()

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
            print("Average of averages (std)", np.mean(averages), np.std(averages))
            print("Average (std)", np.mean(confidences), np.std(confidences))
            print("Max", np.max(confidences))
            print("Min", np.min(confidences))
        corr = pearsonr(all_confidences, all_ratios)
        print("Correlation", corr)


if __name__ == '__main__':
    main()