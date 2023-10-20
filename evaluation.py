from collections import defaultdict
import argparse
import os

import wandb

from utils.data_handling import *
from utils.f1_score import compute_score

# reds = [
#     "P937",
#     "P108",
#     "P488",
#     "P286",
#     "P6",
#     "P39",
#     "P54",
#     "P264",
#     "P551",
#     "P451",
#     "P1308",
#     "P210",
#     "P1037",
# ]


def evaluate(data, predictions, target_mode, prediction_mode, aliases):
    # compute F1 as max across any alias for any answer for the most recent, most frequent, or specific-year answer
    qa_targets, qa_predictions = defaultdict(list), defaultdict(list)
    for query_id, query in data.items():
        relation = query['relation']
        target = list()
        for answer in query['answer']:
            if answer['wikidata_id'] in aliases:
                answer_aliases = aliases[answer['wikidata_id']]
                target += answer_aliases
            target.append(answer['name'])
        if target is None:
            continue
        prediction = get_prediction(predictions, query_id, prediction_mode)
        if not len(prediction["answer"]):
            print("Warning: the prediction for query='{}' was empty.".format(query))
            continue
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
        df, scores = compute_score(predictions=qa_predictions[rel], references=qa_targets[rel])
        yield rel, df, {"n_datapoints": len(qa_targets["all"]), **scores}


def load_queries(data_path):
    unique_queries = dict()
    queries = load_dataset(data_path, split="train")
    for query in queries:
        query_id = "_".join(query['id'].split("_")[:2])
        if query_id not in unique_queries and len(query['answer']):
            unique_queries[query_id] = query
    return unique_queries


def load_aliases(data_path):
    all_aliases = dict()
    aliases = load_dataset(data_path, split="train")
    for qid, al in aliases[0].items():
        all_aliases[qid] = al
    return all_aliases


def main(args):
    experiment_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    project_name = "lm_mutability_preds_eval"
    wandb.init(
        project=project_name,
        name=args.exp_name,
        config=args,
    )

    data = load_queries(args.data_path)
    aliases = load_aliases(args.aliases_path)
    predictions = load_predictions(args.predictions_path)
    for rel, df, scores in evaluate(data, predictions, args.target_mode, args.prediction_mode, aliases):
        df.to_json(os.path.join(experiment_dir, f"{rel}_results_per_example.json"))
        wandb.log({k: v for k, v in scores.items() if not isinstance(v, list)})
        print(f"{rel}: ", scores["ave_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--data_path",
        type=str,
        default="coastalcph/fm_queries",
        help="Path to data",
    )
    parser.add_argument(
        "--aliases_path",
        type=str,
        default="coastalcph/fm_aliases",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    args = parser.parse_args()

    main(args)
