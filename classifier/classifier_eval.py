import argparse
import os
from collections import Counter
from functools import partial

import numpy as np
import torch
import wandb
from datasets import load_dataset
from mdl_classifier import compute_metrics, replace_subject
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)


def main(args, device):
    project_name = "mdl_mutability_classifiers"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.getenv("WANDB_PROJECT")
    run_name = f"(predict-relations) {args.model_name_or_path}"
    if "WANDB_NAME" in os.environ:
        run_name = os.getenv("WANDB_NAME")
    wandb.init(project=project_name, name=run_name, config=args)

    ds = load_dataset("coastalcph/fm_queries")
    ds["all_fm"] = ds["train"]
    relation_to_mutability = {
        r: m for r, m in zip(ds["all_fm"]["relation"], ds["all_fm"]["type"])
    }
    print("Examples per relation", Counter(ds["all_fm"]["relation"]))
    for relation in args.relations:
        rng = np.random.default_rng(int(relation[1:]))
        ds[relation] = ds["all_fm"].filter(lambda ex: ex["relation"] == relation)
        # Select one of the 5 templates at random.
        tuple_id_to_index = {
            s: i
            for i, s in enumerate(list(set([id_[:-2] for id_ in ds[relation]["id"]])))
        }
        assert int(len(ds[relation]) / 5) == len(tuple_id_to_index)
        template_choice = rng.choice(5, int(len(ds[relation]) / 5))
        ds[relation] = ds[relation].filter(
            lambda ex: template_choice[tuple_id_to_index[ex["id"][:-2]]]
            == int(ex["id"][-1])
        )
    ds.pop("all_fm")
    ds = ds.filter(lambda ex: len(ex["answer"]) > 0)
    ds = ds.map(lambda ex: {"label": 1 if ex["type"] == "mutable" else 0})
    if args.clf_mutability == "immutable":
        ds = ds.filter(lambda ex: ex["type"] != "immutable_n")
    elif args.clf_mutability == "immutable_n":
        ds = ds.filter(lambda ex: ex["type"] != "immutable")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenized_ds = ds.map(partial(replace_subject, tokenizer))
    print("Example of training example:", tokenized_ds["train"][0])
    print("Loading model")
    id2label = {1: "MUTABLE", 0: "IMMUTABLE"}
    label2id = {"MUTABLE": 1, "IMMUTABLE": 0}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.output_dir),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    all_metrics = {}
    for relation in args.relations:
        mut_type = relation_to_mutability[relation]
        metrics = trainer.evaluate(
            eval_dataset=tokenized_ds[relation],
            metric_key_prefix=relation,
        )
        metrics[f"{relation}_samples"] = len(tokenized_ds[relation])
        all_metrics.update({f"{mut_type}/{k}": v for k, v in metrics.items()})
        trainer.save_metrics(f"{mut_type}_{relation}", metrics)
    wandb.log(all_metrics)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--clf_mutability",
        choices=["immutable", "immutable_n"],
        required=True,
        type=str,
        help="",
    )
    parser.add_argument("--relations", nargs="+", default=[])
    args = parser.parse_args()
    main(args, device)
