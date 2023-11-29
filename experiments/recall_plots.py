import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifier.mdl_classifier import INSTRUCTION, TEMPLATE_TO_USE
from inference import prepare_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rank(probs, index):
    vals = probs[index]
    return (probs > vals).long().sum().item()


def remove_subj_tokenize(example, prepare_prompt, tokenizer, return_tensors=None):
    query = example["query"].replace("_X_ .", "_X_.")
    text = query.replace("_X_.", "").strip()
    text = prepare_prompt(text).strip()
    return {"text": text, **tokenizer(text, return_tensors=return_tensors)}


def save_macro_averages_plot(df, metric, model_name, output_folder):
    # Macro averages per relation.
    df_mean = (
        df[["mut_type", "layer", f"{metric}_mean"]]
        .groupby(by=["mut_type", "layer"], as_index=False)
        .mean()
    )
    df_std = (
        df[["mut_type", "layer", f"{metric}_std"]]
        .groupby(by=["mut_type", "layer"], as_index=False)
        .mean()
    )
    df_macro_avg = pd.concat([df_mean, df_std[[f"{metric}_std"]]], axis=1)
    df_macro_avg["mean+"] = (
        df_macro_avg[f"{metric}_mean"] + df_macro_avg[f"{metric}_std"]
    )
    df_macro_avg["mean-"] = (
        df_macro_avg[f"{metric}_mean"] - df_macro_avg[f"{metric}_std"]
    )
    colors = {"immutable": "green", "immutable_n": "blue", "mutable": "red"}
    for mut_type in df_macro_avg.mut_type.unique():
        this_mut_df = df_macro_avg[df_macro_avg["mut_type"] == mut_type]
        x = this_mut_df["layer"].values
        plt.plot(
            x,
            this_mut_df[f"{metric}_mean"].values,
            "-o",
            label=mut_type,
            color=colors[mut_type],
            markersize=3.5,
        )
        plt.fill_between(
            x,
            this_mut_df["mean-"].values,
            this_mut_df["mean+"].values,
            alpha=0.1,
            color=colors[mut_type],
        )
    loc = "upper left" if metric == "prob" else "best"
    plt.legend(loc=loc)
    plt.xlabel("layer")
    plt.ylabel(metric)
    plt.grid(linestyle="--")
    plt.title(model_name)
    out_image_file = os.path.join(output_folder, f"{metric}.png")
    plt.savefig(out_image_file, bbox_inches="tight")
    plt.close()


def main(args):
    output_folder = os.path.join(args.output_folder, args.model_name, args.split)
    os.makedirs(output_folder, exist_ok=True)
    wandb.config["final_output_folder"] = output_folder

    dataset_name = "coastalcph/mutability_classifier-1-{}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = []
    for ds_name in [dataset_name.format(i) for i in ["1", "n"]]:
        ds = load_dataset(ds_name)
        tokenized_datasets.append(
            ds.map(
                partial(
                    remove_subj_tokenize,
                    prepare_prompt=lambda q: prepare_prompt(
                        q, args.model_name_or_path, INSTRUCTION, TEMPLATE_TO_USE
                    ),
                    tokenizer=tokenizer,
                    return_tensors="pt",
                )
            )
        )
    print("Example:", tokenized_datasets[0][args.split][0])
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)

    relations = []
    mut_types = []
    ex_id = []
    ranks = []
    probs = []
    embed_matrix = model.lm_head.weight
    with torch.no_grad():
        for tokenized_ds in tokenized_datasets:
            for ex in tqdm(tokenized_ds[args.split]):
                relations.append(ex["relation"])
                mut_types.append(ex["type"])
                ex_id.append(ex["id"])
                outputs = model(
                    input_ids=torch.tensor(ex["input_ids"]).to(device),
                    attention_mask=torch.tensor(ex["attention_mask"]).to(device),
                    output_hidden_states=True,
                )
                pred_token_id = torch.argmax(outputs.logits[0, -1], dim=-1).item()
                probs.append([])
                ranks.append([])
                for hidden_states in outputs.hidden_states:
                    # We only look at the last token predictions.
                    logits_i = embed_matrix @ hidden_states[0, -1]
                    ranks[-1].append(get_rank(logits_i, pred_token_id))
                    probs[-1].append(torch.softmax(logits_i, -1)[pred_token_id].item())
    df = pd.DataFrame(
        [
            (id_, rel, mut, prob[i], rank[i], i)
            for id_, rel, mut, prob, rank in zip(
                ex_id, relations, mut_types, probs, ranks
            )
            for i in range(len(rank))
        ],
        columns=["ex_id", "relation", "mut_type", "prob", "rank", "layer"],
    )
    df.to_json(os.path.join(output_folder, "logits_per_layer.json"))
    df = df.drop(["ex_id"], axis=1)
    # Averages per relation.
    means = df.groupby(by=["relation", "mut_type", "layer"], as_index=False).mean()
    stds = df.groupby(by=["relation", "mut_type", "layer"], as_index=False).std()
    df = means.merge(
        stds,
        on=["relation", "mut_type", "layer"],
        suffixes=["_mean", "_std"].sort_values(by=["layer"]),
    )

    metric = "rank"
    save_macro_averages_plot(df, metric, args.model_name, output_folder)


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
    args = parser.parse_args()

    wandb.init(
        project="recall_per_layer",
        name=" ".join(
            [
                args.model_name,
                args.split,
            ]
        ),
        config=args,
    )
    main(args)
