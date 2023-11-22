from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util.globals import TOKENIZER_TO_PREPEND_SPACE

from .ft_hparams import FTHyperParams


def concat_context_obj(context, obj):
    if not obj:
        return context
    if context[-1] != " " and (not obj or obj[0] != " "):
        return "{} {}".format(context, obj)
    return context + obj


def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix
            print("Update norm:", torch.linalg.vector_norm(upd_matrix).item())

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def get_loss_per_token(logits, input_target, target_ids):
    log_probs = torch.log_softmax(logits, dim=2)
    loss = torch.gather(
        log_probs,
        2,
        torch.where(input_target != -100, input_target, 0).unsqueeze(2),
    ).squeeze(2)
    mask = (input_target != -100).float()
    loss_each = -(loss * mask)
    # Average over multiple tokens.
    loss = loss_each.sum(1) / target_ids.size(0)
    first_token_index = torch.argmax((loss_each != 0).to(dtype=torch.int), dim=-1)
    loss_first_token = loss_each[0, first_token_index]
    loss_each = loss_each[0, first_token_index:]
    return loss_each, loss, loss_first_token


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    # Update target and print info
    request = deepcopy(request)
    if TOKENIZER_TO_PREPEND_SPACE[type(tok)] and request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing FT algo for: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    text = request["prompt"].format(request["subject"])
    target = request["target_new"]["str"]
    old_target = request["old_answer"]["str"]
    target_ids = torch.tensor(tok.convert_tokens_to_ids(tok.tokenize(target))).to(
        "cuda"
    )
    old_target_ids = torch.tensor(
        tok.convert_tokens_to_ids(tok.tokenize(old_target))
    ).to("cuda")

    input_tok = tok(
        concat_context_obj(text, tok.decode(target_ids[:-1])), return_tensors="pt"
    ).to("cuda")
    input_target = torch.tensor(-100, device="cuda").repeat(
        input_tok["input_ids"].shape
    )
    ex_len = input_tok["attention_mask"].sum()
    input_target[0, ex_len - len(target_ids) : ex_len] = target_ids

    input_old_tok = tok(
        concat_context_obj(text, tok.decode(old_target_ids[:-1])), return_tensors="pt"
    ).to("cuda")
    input_old_target = torch.tensor(-100, device="cuda").repeat(
        input_old_tok["input_ids"].shape
    )
    ex_len = input_old_tok["attention_mask"].sum()
    input_old_target[0, ex_len - len(old_target_ids) : ex_len] = old_target_ids

    # Configure optimizer / gradients
    wd = hparams.weight_decay
    print(f"Using weight decay of {wd} for editing {text} -> {target}")
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=wd,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")

        opt.zero_grad()

        logits = model(**input_tok).logits
        loss_each, loss, loss_first_token = get_loss_per_token(
            logits, input_target, target_ids
        )

        with torch.no_grad():
            logits = model(**input_old_tok).logits
            loss_old_each, _, loss_old_first_token = get_loss_per_token(
                logits, input_old_target, old_target_ids
            )

        print(
            f"loss {np.round(loss.item(), 3)} = "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-loss_each).mean().item()}"
            f" / avg prob of [{request['old_answer']['str']}] "
            f"{torch.exp(-loss_old_each).mean().item()}"
        )
        print(
            f"first token prob of [{request['target_new']['str']}] "
            f"{torch.exp(-loss_first_token).item()}"
            f" / first token prob of [{request['old_answer']['str']}] "
            f"{torch.exp(-loss_old_first_token).item()}"
        )

        if loss < 1e-2:
            break

        loss.backward()
        opt.step()

        if type(hparams.norm_constraint) is float:
            eps = hparams.norm_constraint
            with torch.no_grad():
                for k, v in weights.items():
                    v[...] = torch.clamp(
                        v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                    )

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def execute_ft_multiple_requests(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if (
            TOKENIZER_TO_PREPEND_SPACE[type(tok)]
            and request["target_new"]["str"][0] != " "
        ):
            # Space required for correct tokenization
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"]["str"] for r in requests]
    old_targets = [r["old_answer"]["str"] for r in requests]

    # Configure optimizer / gradients
    wd = (
        hparams.weight_decay
        if not isinstance(hparams.wd_power_law, tuple)
        else (len(requests) ** hparams.wd_power_law[0])
        * np.exp(hparams.wd_power_law[1])
    )
    print(f"Using weight decay of {wd} for {len(requests)} edits")
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=wd,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt, old_tgt in zip(
            chunks(texts, hparams.batch_size),
            chunks(targets, hparams.batch_size),
            chunks(old_targets, hparams.batch_size),
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                "cuda"
            )
            old_target_ids = tok(old_tgt, return_tensors="pt", padding=True)[
                "input_ids"
            ].to("cuda")

            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id

            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            probs = torch.nn.functional.log_softmax(
                model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
            )
            loss_each = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                1
            ) / loss_mask.sum(1)
            loss = loss_each.mean()

            with torch.no_grad():
                old_loss_mask = old_target_ids != tok.unk_token_id
                old_loss = -(
                    torch.gather(probs, 1, old_target_ids) * old_loss_mask
                ).sum(1) / loss_mask.sum(
                    1
                )  # Why take this average?
                old_prob = torch.exp(-old_loss).mean().item()

            print(
                f"loss {np.round(loss.item(), 3)} = "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(-loss_each).mean().item()}"
                f" / avg prob of [{request['old_answer']['str']}] {old_prob}"
            )

            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
