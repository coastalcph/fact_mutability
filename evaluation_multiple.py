"Evaluate using multiple templates"
from collections import defaultdict

import argparse
import json

from utils.f1_score import compute_score

from utils.data_handling import *


def evaluate(data, predictions, target_mode, prediction_mode, aliases):
    # compute F1 as max across any alias for any answer for the most recent, most frequent, or specific-year answer
    qa_targets, qa_predictions = defaultdict(list), defaultdict(list)
    for query_id, query in data.items():
        relation = query['relation']
        # target = [a['name'] for a in query['answer']]
        target = list()
        for answer in query['answer']:
            if answer['wikidata_id'] in aliases:
                answer_aliases = aliases[answer['wikidata_id']]
                target += answer_aliases
            target.append(answer['name'])
        if target is None:
            continue
        prediction = get_prediction(predictions, query_id, prediction_mode)
        # if not len(prediction['answer']):
        #    continue
        qa_targets[relation].append(
            {
                "answers": {"answer_start": [0] * len(target), "text": target},
                "id": query_id,
            }
        )
        qa_targets["all"].append(
            {
                "answers": {"answer_start": [0] * len(target), "text": target},
                "id": query_id,
            }
        )
        qa_predictions[relation].append({"prediction_text": prediction["answer"], "id": query_id})
        qa_predictions["all"].append({"prediction_text": prediction["answer"], "id": query_id})

    print("Evaluating on {} datapoints".format(len(qa_targets["all"])))
    for rel in qa_targets.keys():
        yield rel, compute_score(predictions=qa_predictions[rel], references=qa_targets[rel])


def load_queries(data_path):
    queries = dict()
    with open(data_path) as fhandle:
        for line in fhandle:
            query = json.loads(line)
            query_id = query['id']
            if query_id not in queries and len(query['answer']):
                queries[query_id] = query
    return queries


def main(args):
    data = load_queries(args.data_path)
    predictions = load_predictions(args.predictions_path)
    aliases = json.load(open('./data/objects_with_aliases.json'))
    for rel, score in evaluate(data, predictions, args.target_mode, args.prediction_mode, aliases):
        print(rel, score['ave_f1'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/val_with_aliases.json",
        help="Path to data",
    )
    parser.add_argument("--predictions_path", type=str, help="Path to predictions")
    parser.add_argument(
        "--target_mode",
        type=str,
        default="most_recent",
        choices=["most_frequent", "most_recent"],
        help="Which target we evaluate against",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="first_token_probability",
        choices=["perplexity", "first_token_probability"],
        help="Which prediction do we evaluate",
    )
    args = parser.parse_args()

    main(args)
