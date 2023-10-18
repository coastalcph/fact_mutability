import argparse
import os

import wandb

from utils.data_handling import *
from utils.f1_score import compute_score


def evaluate(data, predictions, target_mode, prediction_mode):
    # compute F1 as max across any alias for any answer for the most recent, most frequent, or specific-year answer
    qa_targets, qa_predictions = [], []
    for query in data:
        target = query.get_relevant_target(target_mode)
        if target is None:
            continue
        prediction = get_prediction(predictions, query.id, prediction_mode)
        if not len(prediction["answer"]):
            print("Warning: the prediction for query='{}' was empty.".format(query))
        qa_targets.append(
            {
                "answers": {"answer_start": [0] * len(target), "text": target},
                "id": query.id,
            }
        )
        qa_predictions.append({"prediction_text": prediction["answer"], "id": query.id})

    print("Evaluating on {} datapoints".format(len(qa_targets)))
    df, scores = compute_score(predictions=qa_predictions, references=qa_targets)
    return df, {
        "n_datapoints": len(qa_targets),
        **scores,
    }


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

    data = build_dataset(args.data_path)
    predictions = load_predictions(args.predictions_path)
    df, scores = evaluate(data, predictions, args.target_mode, args.prediction_mode)
    df.to_json(os.path.join(experiment_dir, "results_per_example.json"))
    wandb.log({k: v for k, v in scores.items() if not isinstance(v, list)})
    print("F1: ", scores["ave_f1"])


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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    args = parser.parse_args()

    main(args)
