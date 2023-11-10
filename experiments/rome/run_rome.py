import argparse
import collections
import io
import json
import os
import re
from contextlib import redirect_stdout

import torch
import wandb
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from third_party.rome.rome import ROMEHyperParams, apply_rome_to_model
from third_party.rome.util.globals import HPARAMS_DIR

ROME_UPDATE_HPARAMS = ["v_lr", "v_weight_decay"]


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    if "gpt" not in args.model_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(
            model,
            args.model_name_or_path,
            device_map="auto",
            no_split_module_classes=["LlamaDecoderLayer"],
            offload_folder="./",
        )
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        tok = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not isinstance(model, LlamaForCausalLM),
        )
        print("hf_device_map", model.hf_device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda")
        tok = AutoTokenizer.from_pretrained(args.model_name)

    tok.pad_token = tok.eos_token
    print(model.config)

    requests = [
        {
            "prompt": "{} was the founder of",
            "subject": "Steve Jobs",
            "target_new": {"str": "Microsoft"},
            "old_answer": {"str": "Apple"},
            "seed": args.seed,
        }
    ]

    if args.updates_dataset is not None:
        ds = load_dataset(args.updates_dataset)[args.dataset_split]
        requests = []
        for ex in ds:
            subj = ex["query"]["label"]
            prompt = ex["prediction"]["query"].replace(subj, "{}")
            old_answer = ex["predictions"][0]["answer"]
            for update in ex["updates"]:
                requests.append(
                    {
                        "prompt": prompt,
                        "subject": subj,
                        "target_new": {"str": update},
                        "old_answer": {"str": old_answer},
                        "seed": args.seed,
                    }
                )

    # Execute rewrite
    params_path = os.path.join(
        HPARAMS_DIR, "ROME", f"{args.model_name.replace('/', '_')}.json"
    )
    hparams = ROMEHyperParams.from_json(params_path)
    for hparam_update in ROME_UPDATE_HPARAMS:
        if getattr(args, hparam_update) is not None:
            hparams = getattr(args, hparam_update)
    wandb.config["rome_hparams"] = hparams
    results = []
    for request in tqdm(requests, desc="Requests"):
        print(request)
        output = io.StringIO()
        with redirect_stdout(output):
            model_new, orig_weights = apply_rome_to_model(
                model, tok, [request], hparams, return_orig_weights=True
            )
        print(output.getvalue())

        # Extract data from stdout.
        data = collections.defaultdict(list)
        for line in output.getvalue().split("\n"):
            if line.startswith("loss"):
                data["loss_per_step"].append(float(line[len("loss") : line.find("=")]))
                m = re.match(
                    ".*avg prob of \[.*\] (\d+\.\d+) / avg prob of \[.*\] (\d+\.\d+)",
                    line,
                )
                data["prob_new"].append(float(m.group(1)))
                data["prob_old"].append(float(m.group(2)))
            elif line.startswith("Update norm:"):
                data["update_matrix_norm"].append(float(line[len("Update norm:") :]))

        assert len(data["update_matrix_norm"]) == 1, data["update_matrix_norm"]
        data["request"] = request
        print(data)
        print("----------------------------------------------")
        results.append(data)
    filename = os.path.join(
        args.output_folder, args.exp_name.replace(" ", "_") + ".json"
    )
    with open(filename, "w") as f:
        json.dump(results, f)


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
        "--exp_name",
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
        "--updates_dataset",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "--dataset_split",
        default="validation",
        type=str,
        help="",
    )
    parser.add_argument(
        "--v_lr",
        default=None,
        type=float,
        help="",
    )
    parser.add_argument(
        "--v_weight_decay",
        default=None,
        type=float,
        help="",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=float,
        help="",
    )
    args = parser.parse_args()
    wandb.init(project="rome", name=args.exp_name, config=args)
    torch.manual_seed(args.seed)
    main(args)
