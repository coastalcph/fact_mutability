import os
import json

def main():
    rels_to_replace = [
        "P166",
        "P495",
        "P69",
        "P937"
    ]
    original_files = [
        "alpaca_fmv2_0---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_1---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_2---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_3---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_4---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "llama_0---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_1---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_2---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_3---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_4---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
    ]
    new_files = [
        "alpaca_fmv2_update23Oct_0---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_update23Oct_1---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_update23Oct_2---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_update23Oct_3---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "alpaca_fmv2_update23Oct_4---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B",
        "llama_fmv2_update23Oct_0---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_fmv2_update23Oct_1---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_fmv2_update23Oct_2---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_fmv2_update23Oct_3---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
        "llama_fmv2_update23Oct_4---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B",
    ]

    for og, new in zip(original_files, new_files):
        final_preds = list()
        og_preds = list()
        new_preds = list()
        out = new.replace("update23Oct", "final")
        with open(os.path.join("../fm_predictions/fm_queries_v2", og, "predictions.json")) as fhandle:
            for line in fhandle:
                d = json.loads(line)
                og_preds.append(d)
                rel = d['qcode'].split("_")[1]
                if rel not in rels_to_replace:
                    final_preds.append(d)
        with open(os.path.join("../fm_predictions/fm_queries_v2", new, "predictions.json")) as fhandle:
            for line in fhandle:
                d = json.loads(line)
                new_preds.append(d)
                rel = d['qcode'].split("_")[1]
                if rel in rels_to_replace:
                    final_preds.append(d)
        os.makedirs(os.path.join("../fm_predictions/fm_queries_v2", out), exist_ok=True)
        with open(os.path.join("../fm_predictions/fm_queries_v2", out, "predictions.json"), "w") as fhandle:
            for d in final_preds:
                fhandle.write(f"{json.dumps(d)}\n")


if __name__ == '__main__':
    main()