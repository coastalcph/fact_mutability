import pandas as pd
from collections import defaultdict
import json
import numpy as np
from utils.data_handling import *
from dataset import relations_by_type, sorted_relations_by_type


np.random.seed(42)

def find_query_by_subject(queries, subj):
    for q in queries:
        if q['qid'] == subj:
            return q

def main():
    rels = [r for rels in sorted_relations_by_type.values() for r in rels]
    stats_by_model = defaultdict(lambda: defaultdict(float))
    template_by_model = defaultdict(lambda: defaultdict(int))
    models = [
        ("alpaca", "../fm_predictions/fm_queries_v2/alpaca_fmv2_final_{}---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json"),
        ("llama", "../fm_predictions/fm_queries_v2/llama_fmv2_final_{}---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json"),
        ("llama2", "../fm_predictions/fm_queries_v2/llama2_{}--meta-llama-Llama-2-7b-hf/predictions.json"),
        ("llama_chat", "../fm_predictions/fm_queries_v2/llama2-chat-7b_{}--meta-llama-Llama-2-7b-chat-hf/predictions.json"),
        ("falcon", "../fm_predictions/fm_queries_v2/falcon-7b_no_instr_{}--tiiuae-falcon-7b/predictions.json"),
        ("falcon_instruct", "../fm_predictions/fm_queries_v2/falcon-instruct-7b_{}--tiiuae-falcon-7b-instruct/predictions.json"),
        ("flant5", "../fm_predictions/fm_queries_v2/flant5-xxl_{}--google-flan-t5-xxl/predictions.json"),
        ]
    templates = [0, 1, 2, 3, 4]

    for m, _ in models:
        f1_by_rel = defaultdict(list)
        for t in templates:
            with open(f"./output/{m}_{t}/metrics.jsonl") as fhandle:
                for line in fhandle:
                    rel_metric = json.loads(line)
                    for rel, metric in rel_metric.items():
                        f1_by_rel[rel].append(metric)

        for rel, f1s in f1_by_rel.items():
            if rel != "all":
                stats_by_model[m][rel] = max(f1s)
                template = np.argmax(f1s)
                template_by_model[m][rel] = template

    # alpaca_templates = template_by_model['alpaca']
    for model, pred_file in models:
        print(model)
        model_templates = template_by_model[model]
        for type, rels in relations_by_type.items():
            f1s = list()
            confidences = list()
            for rel in rels:
                template = model_templates[rel]
                stats = pd.read_json(f"./output/{model}_{template}/{rel}_results_per_example.json")
                predictions = load_predictions(pred_file.format(template)) 
                for i, row in stats.iterrows():
                    query_id = row['id']
                    f1 = row['f1']
                    confidence = predictions[query_id]['predictions'][0]['first_token_probability']
                    f1s.append(f1)
                    confidences.append(confidence)
            f1s = np.array(f1s)
            confidences = np.array(confidences)
            bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            digitized = np.digitize(f1s, bins, right=True)
            bin_means = [confidences[digitized == i].mean() for i in range(0, len(bins))]
            bin_std = [confidences[digitized == i].std() for i in range(0, len(bins))]
            bin_counts = [len(confidences[digitized == i]) for i in range(0, len(bins))]
            print(type)
            for m, s, c in zip(bin_means, bin_std, bin_counts):
                print(m, s, c)
            
    

if __name__ == '__main__':
    main()
