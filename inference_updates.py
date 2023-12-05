import argparse
import json
import os

import torch
import wandb
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import collections
from inference import prepare_prompt, get_scores, get_generation_config
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BEAMS = 1
MAX_ANSWER_LENGTH = 10

TEMPLATES = {
    "query_in_instructions": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}: {}\n\n### Response:"
    ),
    "query_in_response": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response: {}"
    ),
    "query_in_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
    ),
}


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

    ds = load_dataset(f"coastalcph/fm-updates-{args.model_name}")["test"]
    templates_ds = load_dataset("coastalcph/fm_templates")["train"]

    outputs = {key: [] for key in ["raw_predictions", "predictions"]}
    updated_counts_mutability = collections.defaultdict(int)
    for ex_i, ex in enumerate(tqdm(ds)):
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
        context = list(templates)[0]
        # TODO: should we run over all?
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
            updated_counts_mutability[f"{ex['type']}_succ"] += 1
        updated_counts_mutability[f"{ex['type']}_total"] += 1
        if ex_i % 100 == 0:
            print(updated_counts_mutability)
            print("query", query)
            print("new_target", new_target)
            print("answer", answer)
    for k, v in updated_counts_mutability.items():
        wandb.run.summary[k] = v

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
    args = parser.parse_args()

    project_name = "prompt_updates"
    wandb.init(
        project=project_name,
        name=" ".join([args.model_name]),
        config=args,
    )

    main(args)
