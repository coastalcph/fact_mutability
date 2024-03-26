import pandas as pd
import os
from collections import defaultdict
import json
import numpy as np
from utils.data_handling import *
from dataset import sorted_relations_by_type
from evaluation import load_aliases


np.random.seed(42)

def find_query_by_subject(queries, subj):
    for q in queries:
        if q['qid'] == subj:
            return q


def concat_aliases(objects, aliases):
    new_objects = list()
    for obj in objects:
        if obj['qid'] in aliases:
            obj['aliases'] = aliases[obj['qid']]
        new_objects.append(obj)
    return new_objects

def main():
    aliases_path = "coastalcph/fm_aliases"
    aliases = load_aliases(aliases_path)
    rels_labels = list()
    for type, rels in sorted_relations_by_type.items():
        for rel in rels:
            rels_labels.append(rel)
        rels_labels.append("Avg")
    rels_labels.append("Average")

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
    for m, pred_file in models:
        print(m)
        model_templates = template_by_model[m]
        for type, rels in sorted_relations_by_type.items():
            print(type)
            queries_for_type = list()
            for rel in rels:
                print(rel)
                template = model_templates[rel]
                selected_queries = list()
                subjects = json.load(open("./data/wikidata/objects_by_freq/{}.json".format(rel)))
                stats = pd.read_json(f"./output/{m}_{template}/{rel}_results_per_example.json")
                predictions = load_predictions(pred_file.format(template))
                good_predictions = stats[stats['f1']==1.0]
                for query_id, f1 in zip(good_predictions['id'].tolist(), good_predictions['f1'].tolist()):
                    pred = predictions[query_id]
                    subj = query_id.split("_")[0]
                    query = find_query_by_subject(subjects, subj)
                    query['objects'] = concat_aliases(query['objects'], aliases)
                    example = {
                        "query": query,
                        "prediction": pred,
                        "f1": f1
                    }
                    selected_queries.append(example)
                confidences = np.array([s['prediction']['predictions'][0]['first_token_probability'] for s in selected_queries])
                bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                digitized = np.digitize(confidences, bins, right=True)
                bin_counts = [len(confidences[digitized == i]) for i in range(0, len(bins))]
                for b, c in zip(bins, bin_counts):
                    print(b, ";", c)
                queries_for_type += selected_queries
                q_high_conf = [s for s in selected_queries if s['prediction']['predictions'][0]['first_token_probability'] >= 0.5]
                filename = f"./data/selected_examples/{m}/"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                json.dump(q_high_conf, open(f"./data/selected_examples/{m}/{rel}.json", "w"), indent=True)

                ### Analysis of selected examples
                print("Num examples", len(selected_queries))
            
            confidences = list()
            for q in queries_for_type:
                confidence = q['prediction']['predictions'][0]['first_token_probability']
                confidences.append(confidence)
            print(f"Average confidence for {type}", np.mean(confidences))
    

if __name__ == '__main__':
    main()
