"""
python train_classifier.py \
    --model_name_or_path "" \
    --output_dir "" \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 8 \
    --weight_decay 0.01 \
    --evaluation_strategy "epoch" \
    --logging_strategy "epoch" \
    --save_strategy "epoch" --save_total_limit 1 \
    --load_best_model_at_end True --do_train
"""
import logging
import os
import sys
from collections import OrderedDict, Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import List, Optional

import evaluate
import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from inference import TEMPLATES, prepare_prompt

logger = logging.getLogger(__name__)
TEMPLATE_TO_USE = "query_in_response"
INSTRUCTION = "Complete the fact in as few words as possible"


@dataclass
class CustomTrainingArguments(TrainingArguments):
    report_to: Optional[List[str]] = field(
        default="wandb",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )
    early_stopping: bool = field(
        default=True, metadata={"help": "Whether to do early stopping"}
    )
    early_stopping_patience: int = field(
        default=4,
        metadata={
            "help": "Number of evaluation steps without improvement before training is terminated."
        },
    )
    do_predict_on_split: Optional[str] = field(
        default="test",
        metadata={
            "help": "The list of integrations to report the results and logs to."
        },
    )


@dataclass
class DataTrainingArguments:
    portion_sizes: List[float]
    portion_idx: int
    dataset_name: Optional[str] = field(
        default="coastalcph/fm_queries_classifier",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    random_labels_per_relation: Optional[bool] = field(default=False)
    random_labels_per_template: Optional[bool] = field(default=False)
    seed_for_random_labels: Optional[int] = field(default=-1)
    subsample_train: Optional[float] = field(default=1.0)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )


def replace_subject(example, prepare_prompt, tokenizer, return_tensors=None):
    query = example["query"].replace("_X_ .", "_X_.")
    text = query.replace("_X_.", example["answer"][0]["name"]).strip()
    text = prepare_prompt(text).strip()
    return {"text": text, **tokenizer(text, return_tensors=return_tensors)}


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    prediction_scores, labels = eval_pred
    predictions = np.argmax(prediction_scores, axis=1)
    exp_pred_scores = np.exp(prediction_scores.astype(np.float64))
    labels_probs = exp_pred_scores[np.arange(len(labels)), labels] / np.sum(
        exp_pred_scores, axis=1
    )
    return {
        **accuracy.compute(predictions=predictions, references=labels),
        **precision.compute(predictions=predictions, references=labels),
        "sum_log2_prob": np.sum(np.log2(labels_probs)),
        "mean_log_prob": np.mean(-np.log(labels_probs)),
    }


def randomize_labels_per_relation(seed, ds):
    rng = np.random.default_rng(seed)
    old_labels = {
        split: {r: l for r, l in zip(ds[split]["relation"], ds[split]["label"])}
        for split in ds.keys()
    }
    # Sorted so we assign the same labels across different runs.
    relations = sorted([r for s in ds.keys() for r in set(ds[s]["relation"])])
    new_labels = {
        relations[i]: label
        for i, label in enumerate(rng.integers(0, 2, len(relations)))
    }
    print("New labels:", new_labels)
    for split in ds.keys():
        changed_labels = sum(
            [int(new_labels[r] != l) for r, l in old_labels[split].items()]
        )
        print("---", split, f"(changed {changed_labels}) ---")
        for r in old_labels[split]:
            if old_labels[split][r] != new_labels[r]:
                print(f"{r} changed {old_labels[split][r]}->{new_labels[r]}")
            else:
                print(f"{r} {new_labels[r]}")
        assert split != "train" or changed_labels > 0
    return ds.map(lambda example: {"label": new_labels[example["relation"]]})


def randomize_labels_per_template(seed, ds):
    def get_relation_template_id_from_id(example_id):
        _, relation, template_id = example_id.split("_")
        return f"{relation}_{template_id}"

    rng = np.random.default_rng(seed)
    old_labels = defaultdict(dict)
    for split in ds.keys():
        for id_, label in zip(ds[split]["id"], ds[split]["label"]):
            old_labels[split][get_relation_template_id_from_id(id_)] = label
    print("old_labels:", old_labels)
    relation_templates = [r_t for s in ds.keys() for r_t in old_labels[s].keys()]
    new_labels = {
        relation_templates[i]: label
        for i, label in enumerate(rng.integers(0, 2, len(relation_templates)))
    }
    for split in ds.keys():
        print("---", split, "---")
        for r_t in sorted(old_labels[split].keys()):
            if old_labels[split][r_t] != new_labels[r_t]:
                print(f"{r_t} changed {old_labels[split][r_t]}->{new_labels[r_t]}")
            else:
                print(f"{r_t} {new_labels[r_t]}")
    return ds.map(
        lambda example: {
            "label": new_labels[get_relation_template_id_from_id(example["id"])]
        }
    )


def main(device):
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    project_name = "mdl_mutability_classifiers"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.getenv("WANDB_PROJECT")
    run_name = "(predict) " if training_args.do_predict else ""
    run_name += model_args.model_name_or_path
    if "WANDB_NAME" in os.environ:
        run_name = os.getenv("WANDB_NAME")
    wandb.init(project=project_name, name=run_name)

    os.makedirs(training_args.output_dir, exist_ok=True)
    if "/" in model_args.model_name_or_path:
        name_path = model_args.model_name_or_path.split("/")
        model_name_for_file = "_".join(name_path[-max(3, len(name_path)) :])
    dirname = "_".join(
        [
            str(data_args.portion_idx),
            model_name_for_file,
            "{:%d%h_%H%M}".format(datetime.today()),
        ]
    )
    training_args.output_dir = os.path.join(training_args.output_dir, dirname)
    os.makedirs(training_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(data_args.dataset_name, use_auth_token=True)
    ds = ds.rename_column("is_mutable", "label").shuffle(seed=training_args.seed)
    if data_args.subsample_train < 1.0:
        print("Going to subsample train, before:", Counter(ds["train"]["relation"]))
        ds["train"] = ds["train"].select(
            np.arange(0, int(len(ds["train"]) * data_args.subsample_train))
        )
        print("After subsample:", Counter(ds["train"]["relation"]))

    seed_for_random_labels = (
        data_args.seed_for_random_labels
        if data_args.seed_for_random_labels != -1
        else training_args.seed
    )
    if data_args.random_labels_per_relation:
        ds = randomize_labels_per_relation(seed_for_random_labels, ds)
    elif data_args.random_labels_per_template:
        ds = randomize_labels_per_template(seed_for_random_labels, ds)

    portion_indices = [
        int(portion_size * 0.01 * len(ds["train"]))
        for portion_size in data_args.portion_sizes
    ]
    # When there is no next batch to evaluate we evaluate on all the training data.
    if data_args.portion_idx + 1 < len(portion_indices):
        ds["train_portion_to_eval"] = ds["train"].select(
            np.arange(
                portion_indices[data_args.portion_idx],
                portion_indices[data_args.portion_idx + 1],
            )
        )
    else:
        ds["train_portion_to_eval"] = ds["train"]
    ds["train_portion_to_train"] = ds["train"].select(
        np.arange(0, portion_indices[data_args.portion_idx])
    )
    print("portion_sizes", data_args.portion_sizes)
    print(f'train_portion_to_train: {len(ds["train_portion_to_train"])}')
    print(f'train_portion_to_eval: {len(ds["train_portion_to_eval"])}')

    tokenized_ds = ds.map(
        partial(
            replace_subject,
            prepare_prompt=lambda q: prepare_prompt(
                q, model_args.model_name_or_path, INSTRUCTION, TEMPLATE_TO_USE
            ),
            tokenizer=tokenizer,
        )
    )
    print("Example of training example:", tokenized_ds["train"][0])
    print("Loading model")
    id2label = {1: "MUTABLE", 0: "IMMUTABLE"}
    label2id = {"MUTABLE": 1, "IMMUTABLE": 0}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train_portion_to_train"],
        eval_dataset={
            "online_eval": tokenized_ds["train_portion_to_eval"],
            "val": tokenized_ds["validation"],
        },
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        ]
        if training_args.early_stopping
        else None,
    )

    print(f'Validation size: {len(tokenized_ds["validation"])}')

    if training_args.do_train:
        for name, param in model.named_parameters():
            if not name.startswith("score"):
                param.requires_grad = False
            print(f"{name} {param.requires_grad}")

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

        print(f"Training/evaluation parameters {training_args}")
        print(f"Data parameters {data_args}")
        print(f"Model parameters {model_args}")

        trainer.train()
        trainer.save_model()
        trainer.save_state()

        print("Training done, evlauating model.")
    for prefix, split in [
        ("f_online_portion", "train_portion_to_eval"),
        ("f_test", "test"),
        ("f_train", "train_portion_to_train"),
        ("f_val", "validation"),
    ]:
        metrics = trainer.evaluate(
            eval_dataset=tokenized_ds[split],
            metric_key_prefix=prefix,
        )
        metrics[f"{prefix}_samples"] = len(tokenized_ds[split])
        trainer.log_metrics(prefix, metrics)
        trainer.save_metrics(prefix, metrics)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)
