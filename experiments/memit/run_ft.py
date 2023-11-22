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
    set_seed,
)

from third_party.memit.baselines.ft import FTHyperParams as HyperParams
from third_party.memit.baselines.ft import execute_ft
from third_party.memit.util.globals import HPARAMS_DIR

UPDATE_HPARAMS = ["lr", "weight_decay"]


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    if "gpt" not in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        tok = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not isinstance(model, LlamaForCausalLM),
        )
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
            old_answer = ex["original_answer"]
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
        HPARAMS_DIR, "FT", f"{args.model_name.replace('/', '_')}.json"
    )
    print("Params path", params_path)
    hparams = HyperParams.from_json(params_path)
    for hparam_update in UPDATE_HPARAMS:
        if getattr(args, hparam_update) is not None:
            setattr(hparams, hparam_update, getattr(args, hparam_update))
    wandb.config["x_hparams"] = hparams
    results = []
    for request in tqdm(requests, desc="Requests"):
        print(request)
        output = io.StringIO()
        with redirect_stdout(output):
            deltas = execute_ft(model, tok, request, hparams)
            for param_name, upd_matrix in deltas.items():
                for n in [1, 2, float("inf")]:
                    print(
                        "Update norm [{}] ({}): {}".format(
                            param_name,
                            n,
                            torch.linalg.vector_norm(upd_matrix, ord=n).item(),
                        )
                    )
        print(output.getvalue())

        # Extract data from stdout.
        data = collections.defaultdict(list)
        for line in output.getvalue().split("\n"):
            if line.startswith("loss"):
                data["loss_per_step"].append(float(line[len("loss") : line.find("=")]))
                m = re.match(
                    ".*avg prob of \[.*\] (\d.*) / avg prob of \[.*\] (\d.*)",
                    line,
                )
                data["prob_new"].append(float(m.group(1)))
                data["prob_old"].append(float(m.group(2)))
            elif line.startswith("first token prob of"):
                m = re.match(
                    "first token prob of \[.*\] (\d.*) / first token prob of \[.*\] (\d.*)",
                    line,
                )
                data["prob_new_token"].append(float(m.group(1)))
                data["prob_old_token"].append(float(m.group(2)))
            elif line.startswith("Update norm"):
                m = re.match(
                    "Update norm \[(.*)\] \((.*)\): (.*)",
                    line,
                )

                data[f"l{m.group(2)}-{m.group(1)}"].append(float(m.group(3)))

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
        "--cache_dir",
        default=None,
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
        "--lr",
        default=None,
        type=float,
        help="",
    )
    parser.add_argument(
        "--weight_decay",
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
    wandb.init(project="ft", name=args.exp_name, config=args)
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    main(args)
