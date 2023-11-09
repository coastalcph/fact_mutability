import collections
import io
import os
import re
from contextlib import redirect_stdout

from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from third_party.rome.rome import ROMEHyperParams, apply_rome_to_model
from third_party.rome.util.globals import HPARAMS_DIR

# MODEL_NAME = "gpt2-medium"
MODEL_NAME = "llama-7b"
ALG_NAME = "ROME"

if "llama" in MODEL_NAME:
    model_name_or_path = "/projects/nlp/data/constanzam/llama/huggingface-ckpts/7B"
    config = AutoConfig.from_pretrained(model_name_or_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model,
        model_name_or_path,
        device_map="auto",
        no_split_module_classes=["LlamaDecoderLayer"],
        offload_folder="./",
    )
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=not isinstance(model, LlamaForCausalLM),
    )
    print(model.hf_device_map)
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

tok.pad_token = tok.eos_token
print(model.config)

requests = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
        "old_answer": {"str": "Apple"},
    }
]

# Execute rewrite
params_path = os.path.join(HPARAMS_DIR, "ROME", f"{MODEL_NAME.replace('/', '_')}.json")
hparams = ROMEHyperParams.from_json(params_path)
output = io.StringIO()
with redirect_stdout(output):
    model_new, orig_weights = apply_rome_to_model(
        model, tok, requests, hparams, return_orig_weights=True
    )
print(output.getvalue())

# Extract data from stdout.
data = collections.defaultdict(list)
for line in output.getvalue().split("\n"):
    if line.startswith("loss"):
        data["loss_per_step"].append(float(line[len("loss") : line.find("=")]))
        m = re.match(
            ".*avg prob of \[.*\] (\d+\.\d+) / avg prob of \[.*\] (\d+\.\d+)", line
        )
        data["prob_new"].append(float(m.group(1)))
        data["prob_old"].append(float(m.group(2)))
    elif line.startswith("Update norm:"):
        data["update_matrix_norm"].append(float(line[len("Update norm:") :]))

assert len(data["update_matrix_norm"]) == 1, data["update_matrix_norm"]
print(data)
