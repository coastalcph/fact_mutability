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
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import List, Optional

import evaluate
import numpy as np
import pandas as pd
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

logger = logging.getLogger(__name__)


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
        default="cfierro/mutability_classifier_data",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )


def replace_subject(tokenizer, example):
    text = re.sub(r" \[Y\]\s?\.?$", "", example["template"].strip())
    text = text.replace("[X]", example["subject"]).strip()
    return tokenizer(text)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    prediction_scores, labels = eval_pred
    predictions = np.argmax(prediction_scores, axis=1)
    best_scores = np.max(prediction_scores, axis=1)
    probs = np.exp(best_scores) / sum(np.exp(best_scores))
    return {
        **accuracy.compute(predictions=predictions, references=labels),
        **precision.compute(predictions=predictions, references=labels),
        "log_prob": np.sum(np.log2(probs)),
    }


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
            data_args.portion_idx,
            model_name_for_file,
            "{:%d%h_%H%M}".format(datetime.today()),
        ]
    )
    training_args.output_dir = os.path.join(training_args.output_dir, dirname)
    os.makedirs(training_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    ds = load_dataset(data_args.dataset_name, use_auth_token=True)
    ds["train"] = (
        ds["train"]
        .rename_column("is_mutable", "label")
        .shuffle(seed=training_args.seed)
    )
    portion_indices = [
        int(portion_size * len(ds["train"])) for portion_size in data_args.portion_sizes
    ]
    if data_args.portion_idx < len(portion_indices) - 1:
        ds["validation"] = ds["train"].select(
            np.arange(
                portion_indices[data_args.portion_idx],
                portion_indices[data_args.portion_idx + 1],
            )
        )
    else:
        ds["validation"] = ds["validation"].rename_column("is_mutable", "label")
    ds["train"] = ds["train"].select(
        np.arange(0, portion_indices[data_args.portion_idx])
    )

    tokenized_ds = ds.map(partial(replace_subject, tokenizer))
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
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

    logger.info(f'Training size: {len(tokenized_ds["train"])}')
    logger.info(f'Validation size: {len(tokenized_ds["validation"])}')

    if training_args.do_train:
        for name, param in model.named_parameters():
            if not name.startswith("score"):
                param.requires_grad = False
            logger.info(f"{name} {param.requires_grad}")

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

        logger.info(f"Training/evaluation parameters {training_args}")
        logger.info(f"Data parameters {data_args}")
        logger.info(f"Model parameters {model_args}")

        trainer.train()
        trainer.save_model()
        trainer.save_state()

        metrics = trainer.evaluate(eval_dataset=tokenized_ds["validation"])
        metrics["eval_samples"] = len(tokenized_ds["validation"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)
