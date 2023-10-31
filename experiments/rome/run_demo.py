import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from third_party.rome.util import nethook
from third_party.rome.util.generate import generate_interactive, generate_fast

from third_party.rome.experiments.py.demo import demo_model_editing

import io
from contextlib import redirect_stdout

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

request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

generation_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
]

# Execute rewrite
output = io.StringIO()
with redirect_stdout(output):
    model_new, orig_weights = demo_model_editing(
        model, tok, request, generation_prompts, alg_name=ALG_NAME
    )
step_losses = []
update_matrix_norm = []
for line in output.getvalue().split("\n"):
    if line.startswith("loss"):
        step_losses.append(float(line[len("loss") : line.find("=")]))
    elif line.startswith("Update norm:"):
        update_matrix_norm.append(float(line[len("Update norm:") :]))

assert len(update_matrix_norm) == 1, len(update_matrix_norm)
print("Captured data:", step_losses)
print(update_matrix_norm[0])

