import collections
import io
import os
import re
from contextlib import redirect_stdout

from transformers import AutoModelForCausalLM, AutoTokenizer

from third_party.rome.rome import ROMEHyperParams, apply_rome_to_model
from third_party.rome.util.globals import HPARAMS_DIR

MODEL_NAME = "gpt2-medium"
ALG_NAME = "ROME"
IS_COLAB = False


model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
        "cuda"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
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
params_path = os.path.join(
    HPARAMS_DIR, "ROME", f"{model.config._name_or_path.replace('/', '_')}.json"
)
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
        data["prob_new"].append(m.group(1))
        data["prob_old"].append(m.group(2))
    elif line.startswith("Update norm:"):
        data["update_matrix_norm"].append(float(line[len("Update norm:") :]))

assert len(data["update_matrix_norm"]) == 1, data["update_matrix_norm"]
print(data)
