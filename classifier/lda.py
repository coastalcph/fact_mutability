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

from classifier.mdl_classifier import INSTRUCTION, TEMPLATE_TO_USE, replace_subject
from inference import prepare_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
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
    X = []
    relations = []
    mut_types = []
    type_to_label = {
        t: i for i, t in enumerate(["immutable", "immutable_n", "mutable"])
    }
    with torch.no_grad():
        for tokenized_ds in tokenized_datasets:
            for ex in tqdm(tokenized_ds[args.split]):
                y.append(type_to_label[ex["type"]])
                relations.append(ex["relation"])
                mut_types.append(ex["type"])
                outputs = model(
                    input_ids=torch.tensor(ex["input_ids"]).to(device),
                    attention_mask=torch.tensor(ex["attention_mask"]).to(device),
                    output_hidden_states=True,
                )
                X.append(outputs.hidden_states[args.layer][0, -1, :].cpu().numpy())
    print("Fitting LDA...")
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    print("explained_variance_ratio_", clf.explained_variance_ratio_)
    wandb.run.summary["explained_variance_ratio_"] = clf.explained_variance_ratio_
    X_transformed = clf.transform(X)

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
        title=f"LDA of classifier {args.split} data (reprs@layer={args.layer})",
    )
    out_image_file = os.path.join(
        args.output_folder, f"{args.model_name}_{args.split}.pdf"
    )
    fig.write_image(out_image_file)
    wandb.config["out_image_file"] = out_image_file


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
        default=-1,
        type=int,
        help="",
    )
    args = parser.parse_args()

    wandb.init(
        project="lda_mutability",
        name=" ".join(
            [
                args.model_name,
                args.split,
            ]
        ),
        config=args,
    )
    main(args)
