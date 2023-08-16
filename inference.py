import argparse
import json
from tqdm import tqdm
import os
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoModelForSeq2SeqLM,
)

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
        "### Instruction:\n{}\n\n### Response:{}"
    ),
    "query_in_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:"
    ),
}


def prepare_prompt(query, args):
    if "alpaca" in args.model_name_or_path:
        instruction = args.instruction
        template = TEMPLATES[args.template]
        return template.format(instruction, query)
    elif "flan" in args.model_name_or_path:
        if len(args.instruction):
            return "{}: {}".format(args.instruction, query)
        else:
            return query
    else:
        return query


def get_scores(model_output, input_ids, prompt, query, tokenizer):
    """Assumes num_beam=1 and is flat-t5 specific. Gets the token scores for every token that is not BOS, EOS or fullstop,
    gets the first non-the token score and computes pplx."""
    sequence = (
        model_output["sequences"][0].cpu().tolist()[1:]
    )  # [1:] ignores the BOS token
    trimmed_sequence = [
        idx for idx in sequence if idx not in [1, 5]
    ]  # ignore EOS and fullstop
    token_scores = [
        torch.softmax(score, 1)[:, idx].cpu().item()
        for idx, score in zip(sequence, model_output["scores"])
        if idx not in [1, 5]
    ]
    answer = tokenizer.decode(trimmed_sequence)
    first_token_score = (
        token_scores[0]
        if not answer.split()[0] in ["the", "a", "an"]
        else token_scores[1]
    )
    perplexity = np.exp(-np.mean(np.log(token_scores)))

    return answer, token_scores, first_token_score, perplexity


def inference(dataset, tokenizer, model, args):
    config = GenerationConfig(
        max_new_tokens=50,
        num_beams=NUM_BEAMS,
        do_sample=False,
        output_hidden_states=False,
        output_scores=False,
        num_return_sequences=NUM_BEAMS,
        return_dict_in_generate=True,
    )

    predictions = []
    outputs = {key: [] for key in ["raw_predictions", "predictions"]}
    for line in tqdm(dataset):
        qcode, query = line.split("\t")
        with torch.no_grad():
            prompt = prepare_prompt(query, args)
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
    print("Loading model")
    if "alpaca" in args.model_name_or_path:
        # the fact tokenizer causes issues with protobuf and tokenizers libraries
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "t5" not in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
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
    experiment_name = "{}--{}".format(
        args.exp_name, args.model_name_or_path.replace("/", "-")
    )
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
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
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="huggyllama/llama-7b",
        help="Model name or path",
    )
    args = parser.parse_args()

    main(args)
