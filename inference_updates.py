import argparse
import collections
import json
import os
from itertools import chain

import numpy as np
import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import get_generation_config, get_scores, prepare_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    experiment_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(experiment_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    use_fast = True
    if (
        "alpaca" in args.model_name_or_path
        or "llama" in args.model_name_or_path.lower()
    ):
        # the fact tokenizer causes issues with protobuf and tokenizers libraries
        use_fast = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=use_fast
    )

    config = get_generation_config(tokenizer)

    ds = load_dataset(f"coastalcph/fm-updates-{args.model_name}")
    templates_ds = load_dataset("coastalcph/fm_templates")["train"]

    outputs = {key: [] for key in ["raw_predictions", "predictions"]}
    updated_counts_mutability = collections.defaultdict(int)
    rng = np.random.default_rng(42)
    for ex_i, ex in enumerate(tqdm(chain(*[ds[split] for split in args.splits]))):
        relation = ex["relation"]
        subject = ex["query"]["label"]
        prompt = ex["prediction"]["query"].replace(subject, "[X]")
        templates = set(
            [
                t.replace("[Y].", "").replace("[Y] .", "").strip()
                for t in templates_ds[relation][0]["templates"]
            ]
        )
        if prompt not in templates:
            print("prompt", prompt)
            print("templates", templates)
            raise Exception("prompt not in templates")
        templates.remove(prompt)
        context = list(templates)[rng.choice(len(templates), 1)[0]]
        new_target = ex["updates"][0]
        query = "Imagine that {} {}. Then, {}".format(
            context.replace("[X]", subject), new_target, prompt.replace("[X]", subject)
        )

        with torch.no_grad():
            prompt = prepare_prompt(
                query, args.model_name_or_path, args.instruction, args.template
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            model_output = model.generate(
                input_ids, generation_config=config, output_scores=True
            )

        answer, token_scores, first_token_score, perplexity = get_scores(
            model_output, input_ids, prompt, query, tokenizer
        )
        outputs["raw_predictions"].append(
            {
                "index": ex_i,
                "query": query,
                "predictions": [
                    {
                        "output_ids": model_output["sequences"][0].cpu().tolist(),
                        "answer": tokenizer.decode(model_output["sequences"][0]),
                    }
                ],
            }
        )
        outputs["predictions"].append(
            {
                "index": ex_i,
                "query": query,
                "new_target": new_target,
                "predictions": [
                    {
                        "answer": answer,
                        "per_token_probability": token_scores,
                        "first_token_probability": first_token_score,
                        "perplexity": perplexity,
                    }
                ],
            }
        )
        if answer.startswith(new_target):
            updated_counts_mutability[f"{ex['type']}/succ"] += 1
        updated_counts_mutability[f"{ex['type']}/total"] += 1
        if ex_i % 100 == 0:
            print(updated_counts_mutability)
            print("query", query)
            print("new_target", new_target)
            print("answer", answer)
    for mutability in list(
        set([k.split("/")[0] for k in updated_counts_mutability.keys()])
    ):
        total = updated_counts_mutability[f"{mutability}_total"]
        succ = updated_counts_mutability[f"{mutability}_succ"]
        wandb.run.summary[f"{mutability}_total"] = total
        wandb.run.summary[f"{mutability}_succ"] = succ
        wandb.run.summary[f"{mutability}_succ_rate"] = succ / total

    print("Writing outputs")
    for key in outputs:
        with open(os.path.join(experiment_dir, key + ".json"), "w") as outfile:
            for i, item in enumerate(outputs[key]):
                outfile.write(json.dumps(item))
                if i != len(outputs[key]) - 1:
                    outfile.write("\n")
    with open(os.path.join(experiment_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--template",
        type=str,
        default="query_in_response",
        help="query_in_instructions, query_in_response or query_in_input",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Complete the fact in as few words as possible",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--model_name", type=str, required=True, help="")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name or path",
    )
    parser.add_argument("--splits", nargs="+", default=["test"], help="")
    args = parser.parse_args()

    project_name = "prompt_updates"
    wandb.init(
        project=project_name,
        name=" ".join([args.model_name]),
        config=args,
    )

    main(args)
