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
from dataclasses import dataclass, field
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
    LlamaModel,
    LlamaPreTrainedModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch import nn

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


class LlamaForSequenceClassificationPerLayer(LlamaPreTrainedModel):
    """This is a copy only changing the hidden_state being used.
    The original code was copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification
    """

    def __init__(self, config, hidden_states_from_layer=-1):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.hidden_states_from_layer = hidden_states_from_layer

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        # hidden_states = transformer_outputs[0]
        hidden_states = transformer_outputs.hidden_states[self.hidden_states_from_layer]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


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

    project_name = "mutability_classifier"
    if "WANDB_PROJECT" in os.environ:
        project_name = os.getenv("WANDB_PROJECT")
    run_name = "(predict) " if training_args.do_predict else ""
    run_name += model_args.model_name_or_path
    if "WANDB_NAME" in os.environ:
        run_name = os.getenv("WANDB_NAME")
    wandb.init(
        project=project_name,
        name=run_name,
        config={**model_args, **data_args, **training_args},
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    if "/" in model_args.model_name_or_path:
        name_path = model_args.model_name_or_path.split("/")
        model_name_for_file = "_".join(name_path[-max(3, len(name_path)) :])
    dirname = "_".join(
        [
            model_name_for_file,
            "{:%d%h_%H%M}".format(datetime.today()),
        ]
    )
    training_args.output_dir = os.path.join(training_args.output_dir, dirname)
    os.makedirs(training_args.output_dir)

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
