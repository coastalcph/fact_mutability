import argparse
import os
import torch, numpy
from collections import defaultdict
from third_party.rome.util import nethook
from third_party.rome.experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    plot_trace_heatmap,
)
from third_party.rome.experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
import numpy as np
from matplotlib import pyplot as plt
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)
from tqdm import tqdm
from datasets import load_dataset
import wandb

torch.set_grad_enabled(False)


def load_model_and_tok(args):
    if args.cache_dir is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, cache_dir=args.cache_dir
        )
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(
            model,
            args.model_name_or_path,
            device_map="auto",
            no_split_module_classes=["LlamaDecoderLayer"],
        )
        print(model.hf_device_map)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not isinstance(model, LlamaForCausalLM),
    )
    return ModelAndTokenizer(
        model_name=args.model_name, model=model, tokenizer=tokenizer
    )


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    """Copy of the function in causal_trace.ipynb"""
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def find_token_range(tokenizer, token_array, subject):
    subj_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(subject))
    token_array = np.array(token_array.cpu())
    for i in range(len(token_array)):
        if i + len(subj_tokens) <= len(token_array) and np.all(
            token_array[i : i + len(subj_tokens)] == subj_tokens
        ):
            return i, i + len(subj_tokens)
    raise Exception(
        "Did not find subj_tokens={} in token_array={}".format(subj_tokens, token_array)
    )


def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Copy of the function in causal_trace.ipynb
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    """Copy of the function in causal_trace.ipynb"""
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    """Copy of the function in causal_trace.ipynb"""
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def plot_last_subj_token(differences, low_score, avg_over_n, modelname, kind, savepdf):
    window = 10
    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
            kind
        ],
        vmin=low_score,
    )
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels(["Last subj token"])
    if not kind:
        ax.set_title("Impact of restoring state after corrupted input")
        ax.set_xlabel(f"single restored layer within {modelname}")
    else:
        kindname = "MLP" if kind == "mlp" else "Attn"
        ax.set_title(
            "Avg. ({}) impact of restoring {} after corrupted input".format(
                avg_over_n, kindname
            )
        )
        ax.set_xlabel(f"Center of interval of {window} restored {kindname} layers")
    plt.colorbar(h)
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main(args):
    data_id = "_".join(args.updates_dataset, args.split, args.mutability_type)
    cache_output_dir = os.path.join(
        args.output_folder, args.model_name, data_id, "cache_hidden_flow"
    )
    pdf_output_dir = os.path.join(args.output_folder, args.model_name, data_id, "plots")
    wandb.config["cache_output_dir"] = cache_output_dir
    wandb.config["plots_output_dir"] = pdf_output_dir
    os.makedirs(cache_output_dir, exist_ok=True)
    os.makedirs(pdf_output_dir, exist_ok=True)
    mt = load_model_and_tok(args)
    print("Testing prediction...")
    print(
        predict_token(
            mt,
            ["Megan Rapinoe plays the sport of", "The Space Needle is in the city of"],
            return_p=True,
        )
    )

    print("Computing noise level...")
    ds = load_dataset(args.updates_dataset)
    ds = ds[args.split]
    if args.mutability_type is not None:
        ds = ds.filter(lambda ex: ex["type"] == "mutability")
    print("Computing causal analysis for", len(ds))
    noise_level = 3 * collect_embedding_std(
        mt,
        [ex["query"]["label"] for ex in ds],
        subjects_from_ds=data_id,
    )
    print(f"Using noise level {noise_level}")
    kind = "mlp"
    for ex in tqdm(ds, desc="Examples"):
        ex_id = f"{ex['query']['rel_id']}_{ex['query']['qid']}"
        filename = os.path.join(cache_output_dir, f"{ex_id}{kind}.npz")
        if not os.path.isfile(filename):
            result = calculate_hidden_flow(
                mt,
                ex["prediction"]["query"],
                ex["query"]["label"],
                noise=noise_level,
                kind=kind,
            )
            numpy_result = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in result.items()
            }
            np.savez(filename, **numpy_result)
        else:
            numpy_result = np.load(filename, allow_pickle=True)
        plot_result = dict(numpy_result)
        plot_result["kind"] = kind
        pdfname = os.path.join(
            pdf_output_dir, f'{str(numpy_result["answer"]).strip()}_{ex_id}{kind}.pdf'
        )
        plot_trace_heatmap(numpy_result, savepdf=pdfname, modelname=args.model_name)

    # Save plot of average.
    scores_last_subj_token = None
    low_scores = 0
    files = os.listdir(cache_output_dir)
    for results_file in files:
        numpy_result = np.load(
            os.path.join(cache_output_dir, results_file), allow_pickle=True
        )
        this_scores = numpy_result["scores"][numpy_result["subject_range"][-1] - 1]
        if scores_last_subj_token is None:
            scores_last_subj_token = this_scores
        else:
            scores_last_subj_token += this_scores
        low_scores += numpy_result["low_score"]
    plot_last_subj_token(
        np.array([scores_last_subj_token / len(files)]),
        low_scores / len(files),
        len(files),
        args.model_name,
        kind,
        savepdf=os.path.join(pdf_output_dir, "avg.pdf"),
    )
    print(
        f"Biggest effect on {kind} on average in layer {np.argmax(scores_last_subj_token)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--model_name_or_path",
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
        "--model_name",
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
        required=True,
        type=str,
        help="",
    )
    parser.add_argument(
        "--split",
        default="validation",
        type=str,
        help="",
    )
    parser.add_argument(
        "--mutability_type",
        default=None,
        type=str,
        help="",
    )
    args = parser.parse_args()
    wandb.init(project="causal_analysis", name=args.model_name, config=args)
    main(args)
