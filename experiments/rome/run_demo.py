from transformers import AutoModelForCausalLM, AutoTokenizer

from third_party.rome.experiments.py.demo import demo_model_editing
from third_party.rome.rome import ROMEHyperParams, apply_rome_to_model
import io
from contextlib import redirect_stdout
from third_party.rome.util.globals import HPARAMS_DIR
import os

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
step_losses = []
update_matrix_norm = []
for line in output.getvalue().split("\n"):
    if line.startswith("loss"):
        step_losses.append(float(line[len("loss") : line.find("=")]))
    elif line.startswith("Update norm:"):
        update_matrix_norm.append(float(line[len("Update norm:") :]))

assert len(update_matrix_norm) == 1, len(update_matrix_norm)
print("Loss per step:", step_losses)
print("Norm of the update:", update_matrix_norm[0])
