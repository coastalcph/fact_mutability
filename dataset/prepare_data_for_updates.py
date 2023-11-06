"""Prepares the data for the updates using the files from analysis/select_examples.py"""
import collections
import numpy as np
import os
import argparse
import json
from datasets import Dataset
from tqdm import tqdm


def main(args):
    rng = np.random.default_rng(7)
    relation_to_examples = collections.defaultdict(list)
    relation_to_answers = collections.defaultdict(set)
    for relation in os.listdir(args.selected_examples_folder):
        with open(os.path.join(args.selected_examples_folder, relation)) as f:
            relation_data = json.load(f)
        for ex in relation_data:
            ex["relation"] = relation[: -len(".json")]
            relation_to_examples[relation].append(ex)
            for ans in ex["prediction"]["predictions"]:
                relation_to_answers[relation].add(ans["answer"])
    relations = sorted(list(relation_to_examples.keys()))
    ds_data = []
    for relation in tqdm(relations, desc="Selecting updates per relation"):
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
                        "Found only {} for an example in relation {}.".format(
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
    ds.push_to_hub("coastalcph/fm_updates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--selected_examples_folder",
        required=True,
        type=str,
        help="",
    )
    args = parser.parse_args()
    main(args)
