"""
python train_classifier.py \
    --model_name_or_path "" \
    --output_dir "" \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 5 \
    --weight_decay 0.01 \
    --evaluation_strategy "epoch" \
    --logging_strategy "epoch" \
    --save_strategy "epoch" --save_total_limit 1 \
    --load_best_model_at_end True --do_train
"""
import logging
import os
import re
import json
import pandas as pd
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import partial
from typing import List, Optional

import evaluate
import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import (
    EarlyStoppingCallback,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from models import (
    LlamaForSequenceClassificationPerLayer,
    T5ForSequenceClassificationPerLayer,
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
    train_classifier_from_layer: int = field(default=-1)


def replace_subject(tokenizer, example):
    text = re.sub(r" \[Y\]\s?\.?$", "", example["template"].strip())
    text = text.replace("[X]", example["subject"]).strip()
    return tokenizer(text)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    roc_auc_score = evaluate.load("roc_auc")
    precision = evaluate.load("precision")
    prediction_scores, labels = eval_pred
    predictions = np.argmax(prediction_scores, axis=1)
    return {
        **accuracy.compute(predictions=predictions, references=labels),
        **roc_auc_score.compute(prediction_scores=predictions, references=labels),
        **precision.compute(predictions=predictions, references=labels),
    }


def init_wandb(model_args, data_args, training_args):
    project_name = "mutability_classifier"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.getenv("WANDB_PROJECT")
    run_name = (
        f"(predict-{training_args.do_predict_on_split}) "
        if training_args.do_predict
        else ""
    )
    run_name += model_args.model_name_or_path
    if "WANDB_NAME" in os.environ:
        run_name = os.getenv("WANDB_NAME")
    wandb.init(
        project=project_name,
        name=run_name,
        config={**asdict(model_args), **asdict(data_args), **asdict(training_args)},
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

    os.makedirs(training_args.output_dir, exist_ok=True)
    if "/" in model_args.model_name_or_path:
        name_path = model_args.model_name_or_path.split("/")
        model_name_for_file = "_".join(name_path[-max(3, len(name_path)) :])
    dirname = "_".join(
        [
            model_name_for_file,
            f"hs_{model_args.train_classifier_from_layer}"
            if training_args.do_train
            else training_args.do_predict_on_split,
            "{:%d%h_%H%M}".format(datetime.today()),
        ]
    )
    training_args.output_dir = os.path.join(training_args.output_dir, dirname)
    os.makedirs(training_args.output_dir)

    init_wandb(model_args, data_args, training_args)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    ds = load_dataset(data_args.dataset_name, use_auth_token=True)
    ds["train"] = ds["train"].rename_column("is_mutable", "label")
    ds["validation"] = ds["validation"].rename_column("is_mutable", "label")
    ds["test"] = ds["test"].rename_column("is_mutable", "label")
    tokenized_ds = ds.map(partial(replace_subject, tokenizer))
    print("Example of training example:", tokenized_ds["train"][0])
    print("Loading model")
    id2label = {1: "MUTABLE", 0: "IMMUTABLE"}
    label2id = {"MUTABLE": 1, "IMMUTABLE": 0}

    if "t5" in model_args.model_name_or_path:
        model = T5ForSequenceClassificationPerLayer.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=True,
            device_map="auto",
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            hidden_states_from_layer=model_args.train_classifier_from_layer,
        ).to(device)
    else:
        model = LlamaForSequenceClassificationPerLayer.from_pretrained(
            model_args.model_name_or_path,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            hidden_states_from_layer=model_args.train_classifier_from_layer,
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

    if training_args.do_train:
        for name, param in model.named_parameters():
            if not name.startswith("score"):
                param.requires_grad = False

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

        logger.info(f"Training/evaluation parameters {training_args}")
        logger.info(f"Data parameters {data_args}")
        logger.info(f"Model parameters {model_args}")

        trainer.train()
        trainer.save_model()
        trainer.save_state()

    if training_args.do_predict:
        trainer_predict = trainer.predict(
            tokenized_ds[training_args.do_predict_on_split], metric_key_prefix="predict"
        )
        pred_score = trainer_predict.predictions
        predictions = np.argmax(pred_score, axis=1)
        metrics = trainer_predict.metrics
        with open(
            os.path.join(training_args.output_dir, f"predict_metrics.txt"), "w"
        ) as writer:
            writer.write(json.dumps(trainer_predict.metrics))
        output_predict_file = os.path.join(
            training_args.output_dir, "predict_results.json"
        )
        if trainer.is_world_process_zero():
            data = []
            for i, pred in enumerate(predictions):
                data.append(
                    (
                        tokenized_ds[training_args.do_predict_on_split][i],
                        tokenized_ds[training_args.do_predict_on_split][i]["relation"],
                        pred,
                        pred_score[i],
                        trainer_predict.label_ids[i],
                    )
                )
            df = pd.DataFrame(
                data, columns=["input", "relation", "prediction", "pred_score", "label"]
            )
            df.to_json(output_predict_file)
            df["correct_pred"] = df["prediction"] == df["label"]
            acc_per_relation = (
                df[["relation", "correct_pred"]]
                .groupby(["relation"], as_index=False)
                .mean()
            )
            for relation, acc in acc_per_relation.values:
                label = df[df.relation == relation].label.unique()[0]
                metrics[f"acc_{relation}_{id2label[label]}"] = acc
            for label, acc in (
                df[["label", "correct_pred"]]
                .groupby(["label"], as_index=False)
                .mean()
                .values
            ):
                metrics[f"acc_{id2label[label]}"] = acc
            wandb.log({f"predict/{k}": v for k, v in metrics.items()})


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)
