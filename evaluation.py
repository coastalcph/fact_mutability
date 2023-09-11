import argparse
import json

from utils.f1_score import compute_score

from utils.data_handling import *


def evaluate(data, predictions, target_mode, prediction_mode):
    # compute F1 as max across any alias for any answer for the most recent, most frequent, or specific-year answer
    qa_targets, qa_predictions = [], []
    for query in data:
        target = query.get_relevant_target(target_mode)
        if target is None:
            continue
        prediction = get_prediction(predictions, query.id, prediction_mode)
        if not len(prediction['answer']):
            continue
        qa_targets.append(
            {
                "answers": {"answer_start": [0] * len(target), "text": target},
                "id": query.id,
            }
        )
        qa_predictions.append({"prediction_text": prediction["answer"], "id": query.id})

    print("Evaluating on {} datapoints".format(len(qa_targets)))
    with open('outputs/eval_targets.json', 'w') as f:
        json.dump(qa_targets, f)
    with open('outputs/eval_predictions.json', 'w') as f:
        json.dump(qa_predictions, f)
    return compute_score(predictions=qa_predictions, references=qa_targets)


def main(args):
    data = build_dataset(args.data_paths, args.dataset, args.dataset_split)
    predictions = load_predictions(args.predictions_path)
    scores = evaluate(data, predictions, args.target_mode, args.prediction_mode)
    print("F1: ", scores["ave_f1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--data_paths",
        nargs="+",
        default=["data/val_with_aliases.json"],
        help="Path to data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default=None,
        help="",
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
