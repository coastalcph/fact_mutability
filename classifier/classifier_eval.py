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
    rng = np.random.default_rng(7)
    print("Examples per relation", Counter(ds["all_fm"]["relation"]))
    for relation in args.relations:
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
    ds = ds.map(lambda ex: {"is_mutable": 1 if ex["type"] == "mutable" else 0})
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
        train_dataset=tokenized_ds["train_portion_to_train"],
        eval_dataset={
            "online_eval": tokenized_ds["train_portion_to_eval"],
            "val": tokenized_ds["validation"],
        },
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    for relation in args.relations:
        metrics = trainer.evaluate(
            eval_dataset=tokenized_ds[relation],
            metric_key_prefix=relation,
        )
        metrics[f"{relation}_samples"] = len(tokenized_ds[relation])
        trainer.log_metrics(relation, metrics)
        trainer.save_metrics(relation, metrics)


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
        "--clf_mutability",
        choices=["immutable", "immutable_n"],
        required=True,
        type=str,
        help="",
    )
    parser.add_argument("--relations", nargs="+", default=[])
    args = parser.parse_args()
    main(args, device)
