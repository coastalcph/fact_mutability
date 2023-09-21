""" Official evaluation script for v1.1 of the SQuAD dataset. """

import re
import string
import sys
from collections import Counter

import pandas as pd


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    gt = normalize_answer(ground_truth)
    prediction_tokens = normalize_answer(prediction)[:len(gt)].split()
    ground_truth_tokens = gt.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    gt = normalize_answer(ground_truth)
    return normalize_answer(prediction)[:len(gt)] == gt


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def _compute(dataset, predictions):
    f1 = []
    exact_match = []
    df_data = []
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa["id"] not in predictions:
                    message = (
                        "Unanswered question " + qa["id"] + " will receive score 0."
                    )
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match.append(
                    metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths
                    )
                )
                f1.append(
                    metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                )
                df_data.append(
                    (qa["id"], prediction, ground_truths, f1[-1], exact_match[-1])
                )

    ave_exact_match = 100.0 * sum(exact_match) / len(exact_match)
    ave_f1 = 100.0 * sum(f1) / len(f1)

    return pd.DataFrame(
        df_data, columns=["id", "prediction", "ground_truth", "f1", "exact_match"]
    ), {
        "exact_match": exact_match,
        "f1": f1,
        "ave_exact_match": ave_exact_match,
        "ave_f1": ave_f1,
    }


def compute_score(predictions, references):
    pred_dict = {
        prediction["id"]: prediction["prediction_text"] for prediction in predictions
    }
    dataset = [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "answers": [
                                {"text": answer_text}
                                for answer_text in ref["answers"]["text"]
                            ],
                            "id": ref["id"],
                        }
                        for ref in references
                    ]
                }
            ]
        }
    ]
    return _compute(dataset=dataset, predictions=pred_dict)
