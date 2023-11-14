"""Prepares the data for the updates using the files from analysis/select_examples.py"""
import argparse
import collections
import json
import os
import re

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from evalutation.f1_score import f1_score

SEED = 7


def get_truncated_ans(func, ground_truths, pred):
    best_score, best_gt = 0, None
    for gt in ground_truths:
        score = func(pred, gt)
        if score > best_score:
            best_score, best_gt = score, gt
    if best_gt is None:
        print(
            "All groundtruths scored 0 f1_score. Skipping example.".format(
                ground_truths, pred
            )
        )
        return None
    if best_score < 1.0:
        print(
            "Warning: the best f1 score found was less than 1.0 ({}), gt={} pred={}".format(
                best_score, ground_truths, pred
            )
        )
        return None  # not necessary, but to debug
    index_found = pred.lower().find(best_gt.lower().split()[-1])
    if index_found == -1:
        print(
            "Warning: did not find exact match: '{}' not in '{}'".format(best_gt, pred)
        )
        return None
    return pred[: index_found + len(best_gt)]


def main(args):
    fm_queries_ds = load_dataset("coastalcph/fm_queries")["train"]
    relation_to_mut_type = {
        r: m for r, m in zip(fm_queries_ds["relation"], fm_queries_ds["type"])
    }
    rng = np.random.default_rng(SEED)
    relation_to_examples = collections.defaultdict(list)
    relation_to_answers = collections.defaultdict(set)
    for relation in os.listdir(args.selected_examples_folder):
        with open(os.path.join(args.selected_examples_folder, relation)) as f:
            relation_data = json.load(f)
        for ex in relation_data:
            ex["relation"] = relation[: -len(".json")]
            ex["type"] = relation_to_mut_type[ex["relation"]]
            relation_to_examples[relation].append(ex)
            possible_correct_answers = [o["label"] for o in ex["query"]["objects"]]
            possible_correct_answers.extend(
                [
                    a
                    for o in ex["query"]["objects"]
                    if "aliases" in o
                    for a in o["aliases"]
                ]
            )
            ans = ex["prediction"]["predictions"][0]["answer"]
            pred_truncated = get_truncated_ans(f1_score, possible_correct_answers, ans)
            if pred_truncated is None:
                print("bug?", ex["query"]["qid"])
                continue
            relation_to_answers[relation].add(pred_truncated.replace("\n", " "))
            ex["original_answer"] = pred_truncated
    relations = sorted(list(relation_to_examples.keys()))

    # Select object updates at random for each example.
    ds_data = []
    for relation in tqdm(relations, desc="Selecting updates per relation"):
        if len(relation_to_examples[relation]) == 1:
            print(
                "Warning: there is only one example for relation {}, skipping relation.".format(
                    relation
                )
            )
            continue
        for ex in relation_to_examples[relation]:
            ex["updates"] = []
            updates_words = set()
            model_answer = set(
                ex["prediction"]["predictions"][0]["answer"].lower().split(" ")
            )
            possible_answers = [
                ans
                for ans in list(relation_to_answers[relation])
                if model_answer.difference(ans.lower().split(" "))
            ]
            if not possible_answers:
                print(
                    "Warning: all answers are the same in relation {}, skipping relation.".format(
                        relation
                    )
                )
                continue
            while (len(ex["updates"])) < 3:
                if not possible_answers:
                    print(
                        "Found only {} updates for an example in relation {}.".format(
                            len(ex["updates"]), relation
                        )
                    )
                    break
                chosen = rng.choice(len(possible_answers), 1)[0]
                ex["updates"].append(possible_answers[chosen])
                updates_words.update(set(possible_answers[chosen].lower().split(" ")))
                possible_answers = [
                    ans
                    for ans in list(relation_to_answers[relation])
                    if set(ans.lower().split(" ")).difference(updates_words)
                ]
            ds_data.append(ex)
    ds = Dataset.from_list(ds_data)
    print(ds)
    print(collections.Counter([f"{r}-{t}" for r, t in zip(ds["relation"], ds["type"])]))
    mut_type_counts = collections.Counter(ds["type"])
    print(mut_type_counts)

    # Split validation set.
    validation_count = max(1, int(min(list(mut_type_counts.values())) * 0.1))
    validation_indices = []
    for mut_type in mut_type_counts.keys():
        ds_mut_type = ds.filter(lambda ex: ex["type"] == mut_type)
        relations = sorted(set(ds_mut_type["relation"]))
        count_per_relation = int(validation_count / len(relations))
        count_left = validation_count - count_per_relation * len(relations)
        count_per_relation = [count_per_relation for _ in relations]
        for i in range(count_left):
            count_per_relation[i] += 1
        assert sum(count_per_relation) == validation_count
        for relation_i, relation in enumerate(relations):
            rel_indices = [i for i, ex in enumerate(ds) if ex["relation"] == relation]
            validation_indices.extend(
                rng.choice(rel_indices, count_per_relation[relation_i], replace=False)
            )
    train_indices = list(set(list(range(len(ds)))).difference(validation_indices))
    ds_splitted = DatasetDict(
        {
            "test": ds.select(indices=train_indices),
            "validation": ds.select(indices=validation_indices),
        }
    )
    print(ds_splitted)
    for split in ds_splitted.keys():
        print(split)
        print(
            collections.Counter(
                [
                    f"{r}-{t}"
                    for r, t in zip(
                        ds_splitted[split]["relation"],
                        ds_splitted[split]["type"],
                    )
                ]
            )
        )
        print(collections.Counter(ds_splitted[split]["type"]))
    if args.hf_dataset_name is not None:
        ds_splitted.push_to_hub(args.hf_dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--selected_examples_folder",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--hf_dataset_name",
        default=None,
        type=str,
        help="",
    )
    args = parser.parse_args()
    main(args)
