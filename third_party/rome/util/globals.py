from pathlib import Path

import yaml
from transformers import (
    GPT2TokenizerFast,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    T5TokenizerFast,
)

with open("third_party/rome/globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(
    RESULTS_DIR,
    DATA_DIR,
    STATS_DIR,
    HPARAMS_DIR,
) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]

TOKENIZER_TO_PREPEND_SPACE = {
    LlamaTokenizerFast: False,
    GPT2TokenizerFast: True,
    T5TokenizerFast: True,
    LlamaTokenizer: False,
    GPTNeoXTokenizerFast: True,
}
