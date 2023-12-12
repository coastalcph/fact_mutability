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
    PreTrainedTokenizerFast,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BEAMS = 1
MAX_ANSWER_LENGTH = 10
DEF_TEMPLATE_TO_USE = "query_in_response"
DEF_INSTRUCTION = "Complete the fact in as few words as possible"

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
    elif "instruct" in model_name_or_path:
        return "{}\n{}".format(instruction, query)
    elif "chat" in model_name_or_path:
        return "[INST] {}: {} [/INST] ".format(instruction, query)
    else:
        return query


get_sequence = {
    # Ignore the prompt.
    LlamaTokenizer: lambda seq, input_ids: seq[input_ids.shape[1] :].cpu().tolist(),
    PreTrainedTokenizerFast: lambda seq, input_ids: seq[input_ids.shape[1] :]
    .cpu()
    .tolist(),
    # Ignore the BOS token.
    T5TokenizerFast: lambda seq, _: seq.cpu().tolist()[1:],
}
ids_to_ignore = {
    # Ignore BOS, EOS.
    LlamaTokenizer: [1, 2],
    # Ignore EOS.
    T5TokenizerFast: [1],
    # Ignore EOS.
    PreTrainedTokenizerFast: [11],
}
# Token id of a full stop when not at the beggining of a word so it could be
# different than tokenizer.tokens_to_ids(tokenizer.tokenize('.')).
full_stop = {LlamaTokenizer: 29889, T5TokenizerFast: 5, PreTrainedTokenizerFast: 25}


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
    answer = tokenizer.decode(trimmed_sequence).strip()
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


def get_generation_config(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return GenerationConfig(
        max_new_tokens=50,
        num_beams=NUM_BEAMS,
        do_sample=False,
        output_hidden_states=False,
        output_scores=False,
        num_return_sequences=NUM_BEAMS,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )


def inference(dataset, tokenizer, model, args):
    config = get_generation_config(tokenizer)
    predictions = []
    outputs = {key: [] for key in ["raw_predictions", "predictions"]}
    for line in tqdm(dataset):
        qcode, query = line.split("\t")
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
                "qcode": qcode,
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
                "qcode": qcode,
                "query": query,
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
    return outputs


def main(args):
    experiment_name = "{}--{}".format(
        args.exp_name, args.model_name_or_path.replace("/", "-")
    )
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print("Loading model")
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
    if "t5" not in args.model_name_or_path:
        if args.cache_dir is not None:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(
                device
            )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, load_in_8bit=True, device_map="auto"
        )
    model.eval()

    print("Loading dataset")
    dataset = open(args.queries_path).read().strip().split("\n")

    print("Running inference")
    outputs = inference(dataset, tokenizer, model, args)

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
        "--queries_path",
        type=str,
        default="data/templama/val.txt",
        help="Path to txt file, one query per line",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEF_TEMPLATE_TO_USE,
        help="query_in_instructions, query_in_response or query_in_input",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEF_INSTRUCTION,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Dir where model outputs will be stored",
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="huggyllama/llama-7b",
        help="Model name or path",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="",
    )
    args = parser.parse_args()

    project_name = "lm_mutability_preds_eval"
    wandb.init(
        project=project_name,
        name="(inference) " + args.exp_name,
        config=args,
    )

    main(args)
