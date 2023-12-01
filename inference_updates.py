import argparse
import json
import os

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
    T5TokenizerFast,
)
import collections
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


def prepare_prompt(query, model_name_or_path, instruction, template=None):
    if "alpaca" in model_name_or_path:
        instruction = instruction
        template = TEMPLATES[template]
        return template.format(instruction, query)
    elif "flan" in model_name_or_path:
        if len(args.instruction):
            return "{}: {}".format(instruction, query)
        else:
            return query
    elif "chat" in model_name_or_path:
        return "[INST] {}: {} [/INST] ".format(instruction, query)
    else:
        return query


get_sequence = {
    # Ignore the prompt.
    LlamaTokenizer: lambda seq, input_ids: seq[input_ids.shape[1] :].cpu().tolist(),
    # Ignore the BOS token.
    T5TokenizerFast: lambda seq, _: seq.cpu().tolist()[1:],
}
ids_to_ignore = {
    # Ignore BOS, EOS.
    LlamaTokenizer: [1, 2],
    # Ignore EOS.
    T5TokenizerFast: [1],
}
# Token id of a full stop when not at the beggining of a word so it could be
# different than tokenizer.tokens_to_ids(tokenizer.tokenize('.')).
full_stop = {
    LlamaTokenizer: 29889,
    T5TokenizerFast: 5,
}


def get_scores(model_output, input_ids, prompt, query, tokenizer):
    """Assumes num_beam=1. Gets the token scores for every token that is not BOS, EOS or fullstop,
    gets the first non-the token score and computes pplx."""
    sequence = get_sequence[type(tokenizer)](model_output["sequences"][0], input_ids)
    assert len(sequence) == len(model_output["scores"])
    token_scores = []
    trimmed_sequence = []
    for idx, score in zip(sequence, model_output["scores"]):
        if idx not in ids_to_ignore[type(tokenizer)]:
            token_scores.append(torch.softmax(score, 1)[:, idx].cpu().item())
            trimmed_sequence.append(idx)
    if trimmed_sequence and trimmed_sequence[-1] == full_stop[type(tokenizer)]:
        token_scores = token_scores[:-1]
        trimmed_sequence = trimmed_sequence[:-1]
    answer = tokenizer.decode(trimmed_sequence)
    words = answer.split()
    if (
        not token_scores
        or not words
        or (
            (len(token_scores) == 1 or len(words) == 1)
            and words[0] in ["the", "a", "an"]
        )
    ):
        print(
            "Warning: Empty generation. input_ids={}, output_sequence={}".format(
                input_ids, sequence
            )
        )
        return "", [], 0, float("inf")
    first_token_score = (
        token_scores[1] if words[0] in ["the", "a", "an"] else token_scores[0]
    )
    perplexity = np.exp(-np.mean(np.log(token_scores)))

    return answer, token_scores, first_token_score, perplexity


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

    config = GenerationConfig(
        max_new_tokens=50,
        num_beams=NUM_BEAMS,
        do_sample=False,
        output_hidden_states=False,
        output_scores=False,
        num_return_sequences=NUM_BEAMS,
        return_dict_in_generate=True,
    )

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

