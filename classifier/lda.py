import argparse
import os
from functools import partial

import pandas as pd
import plotly.express as px
import torch
import wandb
from datasets import load_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import numpy as np
import collections
from classifier.mdl_classifier import INSTRUCTION, TEMPLATE_TO_USE, replace_subject
from inference import prepare_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def randomize_labels_per_relation(y, relations, seed):
    rng = np.random.default_rng(seed)
    old_labels = {r: label for r, label in zip(relations, y)}
    # Sorted so we assign the same labels across different runs.
    relations = sorted(relations)
    new_labels = {
        relations[i]: label
        for i, label in enumerate(rng.integers(0, 3, len(relations)))
    }
    print("New labels:", new_labels)
    print(collections.Counter(new_labels.values()))
    changed_labels = sum(
        [int(new_labels[r] != old_label) for r, old_label in old_labels.items()]
    )
    print("changed labels:", changed_labels)
    for r in old_labels:
        if old_labels[r] != new_labels[r]:
            print(f"{r} changed {old_labels[r]}->{new_labels[r]}")
        else:
            print(f"{r} {new_labels[r]}")
    return [new_labels[r] for r, _ in zip(relations, y)]


def get_hidden_states_repr(args):
    dataset_name = "coastalcph/mutability_classifier-1-{}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = []
    for ds_name in [dataset_name.format(i) for i in ["1", "n"]]:
        ds = load_dataset(ds_name)
        tokenized_datasets.append(
            ds.map(
                partial(
                    replace_subject,
                    prepare_prompt=lambda q: prepare_prompt(
                        q, args.model_name_or_path, INSTRUCTION, TEMPLATE_TO_USE
                    ),
                    tokenizer=tokenizer,
                    return_tensors="pt",
                )
            )
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)

    y = []
    X = collections.defaultdict(list)
    relations = []
    mut_types = []
    type_to_label = {
        t: i for i, t in enumerate(["immutable", "immutable_n", "mutable"])
    }
    for tokenized_ds in tokenized_datasets:
        for ex in tqdm(tokenized_ds[args.split]):
            y.append(type_to_label[ex["type"]])
            relations.append(ex["relation"])
            mut_types.append(ex["type"])
            with torch.no_grad():
                outputs = model(
                    input_ids=torch.tensor(ex["input_ids"]).to(device),
                    attention_mask=torch.tensor(ex["attention_mask"]).to(device),
                    output_hidden_states=True,
                )
            layers = (
                [args.layer]
                if args.layer is not None
                else range(len(outputs.hidden_states))
            )
            for layer in layers:
                X[f"layer={layer}"].append(
                    outputs.hidden_states[layer][0, -1, :].cpu().numpy()
                )
    return X, y, relations, mut_types


def main(args):
    output_folder = os.path.join(args.output_folder, args.model_name, args.split)
    os.makedirs(output_folder, exist_ok=True)
    wandb.config["final_output_folder"] = output_folder
    cache_filename = os.path.join(output_folder, "hidden_states_per_layer.npz")
    if os.path.exists(cache_filename):
        print("Loading data from cache file", cache_filename)
        numpy_result = dict(np.load(cache_filename, allow_pickle=True))
        X, y, relations, mut_types = (
            {k: numpy_result[k] for k in numpy_result.keys() if k.startswith("layer=")},
            numpy_result["y"],
            numpy_result["relations"],
            numpy_result["mut_types"],
        )
    else:
        X, y, relations, mut_types = get_hidden_states_repr(args)
        np.savez(
            cache_filename,
            **{**X, "y": y, "relations": relations, "mut_types": mut_types},
        )
    if args.random_labels:
        y = randomize_labels_per_relation(y, relations, args.random_labels_seed)
        args.random_labels = args.random_labels_seed
        assert len(set(y)) == 3

    layers = [f"layer={args.layer}"] if args.layer is not None else X.keys()
    for layer in layers:
        cache_filename = os.path.join(
            output_folder, f"X_transformed_{layer}_rand={args.random_labels}.npz"
        )
        if os.path.exists(cache_filename):
            print("Loading cached X_transformed from", cache_filename)
            X_transformed = dict(np.load(cache_filename, allow_pickle=True))["arr_0"]
        else:
            print("Fitting LDA...", layer)
            clf = LinearDiscriminantAnalysis()
            clf.fit(X[layer], y)
            print("explained_variance_ratio_", clf.explained_variance_ratio_)
            wandb.run.summary[
                f"explained_variance_ratio_{layer}"
            ] = clf.explained_variance_ratio_
            X_transformed = clf.transform(X[layer])
            np.savez(
                cache_filename,
                X_transformed,
            )

        df = pd.DataFrame(
            zip(X_transformed[:, 0], X_transformed[:, 1], mut_types, relations),
            columns=["1st. dim", "2nd. dim", "mut_type", "relation"],
        )
        fig = px.scatter(
            df,
            x="1st. dim",
            y="2nd. dim",
            color="relation",
            symbol="mut_type",
            symbol_sequence=[0, "diamond-open", 304],
            title=f"LDA of classifier {args.split} data (reprs@{layer})",
        )
        out_image_file = os.path.join(
            output_folder, f"lda_layer{layer}_rand={args.random_labels}.pdf"
        )
        fig.write_image(out_image_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--split",
        default="train",
        type=str,
        help="",
    )
    parser.add_argument(
        "--layer",
        default=None,
        type=int,
        help="",
    )
    parser.add_argument("--random_labels", action="store_true")
    parser.add_argument("--random_labels_seed", default=7)
    args = parser.parse_args()

    wandb.init(
        project="lda_mutability",
        name=" ".join(
            [args.model_name, args.split, "rand" if args.random_labels else ""]
        ),
        config=args,
    )
    main(args)