import pandas as pd
from collections import defaultdict
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from python2latex import Document, Table, bold
from mutability.domain import Relation
import os
from utils.data_handling import *
from dataset import relations, sorted_relations_by_type


np.random.seed(42)
EVAL_DIR = "./output/"


def build_relations(dataset):
    relations_mutations = defaultdict(list)
    for query in dataset:
        relations_mutations[query.relation_id].append(query.get_ratio())
    relations = dict()
    for relation, mutations in relations_mutations.items():
        mean = np.mean(mutations)
        std = np.std(mutations)
        relation_obj = Relation(relation, mean, std)
        relations[relation] = relation_obj
    return relations


def get_acc_per_type(stats_by_model):
    ds = load_dataset("coastalcph/fm_queries")["train"]
    relation_to_type = {ex["relation"]: ex["type"] for ex in ds}
    records = []
    for model, relation_to_acc in stats_by_model.items():
        for relation, acc in relation_to_acc.items():
            records.append(
                {
                    "model": model,
                    "relation": relation,
                    "type": relation_to_type[relation],
                    "acc": acc,
                }
            )
    df = pd.DataFrame(records)
    return (
        df[["model", "type", "acc"]]
        .groupby(by=["model", "type"], as_index=False)
        .mean()
        .sort_values(by=["type"])
    )


def main():
    rels_labels = list()
    for type, rels in sorted_relations_by_type.items():
        for rel in rels:
            rels_labels.append(rel)
        rels_labels.append("Avg")
    rels_labels.append("Average")
    rels = [r for rels in sorted_relations_by_type.values() for r in rels]
    stats_by_model = defaultdict(lambda: defaultdict(float))
    template_by_model = defaultdict(lambda: defaultdict(int))
    models = [f[:-2] for f in os.listdir(EVAL_DIR)]
    templates = [0, 1, 2, 3, 4]
    for m in models:
        f1_by_rel = defaultdict(list)
        for t in templates:
            with open(os.path.join(EVAL_DIR, f"{m}_{t}", "metrics.jsonl")) as fhandle:
                for line in fhandle:
                    rel_metric = json.loads(line)
                    for rel, metric in rel_metric.items():
                        f1_by_rel[rel].append(metric)
        for rel, f1s in f1_by_rel.items():
            if rel != "all":
                stats_by_model[m][rel] = max(f1s)
                template = np.argmax(f1s)
                template_by_model[m][rel] = template

    col = len(stats_by_model) + 2
    row = len(stats_by_model["alpaca"].keys()) + 2 + len(sorted_relations_by_type) + 1
    doc = Document(
        filename="f1_scores",
        filepath="output/tables",
        doc_type="standalone",
        border="10pt",
    )
    table = doc.new(
        Table(shape=(row, col), as_float_env=False, alignment=["c", "l", "c", "c"])
    )
    table[0, 2:] = "Models"
    table[0, 2:].apply_command(bold)
    table[1, 1] = "Relation"
    table[1, 2:] = [m.capitalize() for m in stats_by_model.keys()]
    table[1, 1:].apply_command(bold)
    table[2:, 1] = rels_labels
    for i, model in enumerate(models):
        data = list()
        for type, rels in sorted_relations_by_type.items():
            f1s = list()
            for rel in rels:
                f1 = stats_by_model[model][rel]
                data.append(f1)
                f1s.append(f1)
            data.append(np.mean(f1s))
        data.append(np.mean(data))
        table[2:, i + 2] = data

    # Header rule
    table[0, 2:].add_rule()
    #  rule
    table[1, 1:].add_rule()
    for r in range(2, row + 2):
        table[r].highlight_best("high", "bold")

    tex = doc.build(compile_to_pdf=False, show_pdf=False)
    print(tex)

    all_num_objects = defaultdict(lambda: defaultdict(list))
    all_confidences = defaultdict(lambda: defaultdict(list))
    models = [
        (
            "alpaca",
            "../fm_predictions/fm_queries_v2/alpaca_fmv2_final_{}---projects-nlp-data-constanzam-stanford_alpaca-huggingface-ckpts-7B/predictions.json",
        ),
        (
            "llama",
            "../fm_predictions/fm_queries_v2/llama_fmv2_final_{}---projects-nlp-data-constanzam-llama-huggingface-ckpts-7B/predictions.json",
        ),
        (
            "llama2",
            "../fm_predictions/fm_queries_v2/llama2_{}--meta-llama-Llama-2-7b-hf/predictions.json",
        ),
        (
            "llama_chat",
            "../fm_predictions/fm_queries_v2/llama2-chat-7b_{}--meta-llama-Llama-2-7b-chat-hf/predictions.json",
        ),
        (
            "falcon",
            "../fm_predictions/fm_queries_v2/falcon-7b_no_instr_{}--tiiuae-falcon-7b/predictions.json",
        ),
        (
            "falcon_instruct",
            "../fm_predictions/fm_queries_v2/falcon-instruct-7b_{}--tiiuae-falcon-7b-instruct/predictions.json",
        ),
        (
            "flant5",
            "../fm_predictions/fm_queries_v2/flant5-xxl_{}--google-flan-t5-xxl/predictions.json",
        ),
    ]
    for m, k in models:
        for relation, cls in relations.items():
            t = template_by_model[m][relation]
            predictions = load_predictions(k.format(t))  # LAMA
            subjects = json.load(
                open("./data/wikidata/objects_by_freq/{}.json".format(relation))
            )
            stats = pd.read_json(
                f"./output/{m}_{t}/{relation}_results_per_example.json"
            )
            for subject in subjects:
                subj_label = subject["label"]
                subj_id = subject["qid"]
                prediction = predictions[f"{subj_id}_{relation}"]
                confidences = sorted(
                    [p["first_token_probability"] for p in prediction["predictions"]],
                    reverse=True,
                )
                confidences = [c for c in confidences if not np.isnan(c)]
                if len(confidences) and len(subject["objects"]):
                    confidence = confidences[0]
                    num_objects = len(subject["objects"])
                    all_confidences[m][cls].append(confidence)
                    all_num_objects[m][cls].append(num_objects)
            # import pdb; pdb.set_trace()

    # corr = pearsonr(all_confidences, all_num_objects)
    # print("Correlation", corr)

    for m in all_num_objects.keys():
        print(m)
        for cls in all_num_objects[m].keys():
            print(
                cls, np.mean(all_confidences[m][cls]), np.std(all_confidences[m][cls])
            )
        print()

    for m in all_num_objects.keys():
        fig = plt.figure()
        ax1 = fig.add_subplot()
        colors = {"immutable": "green", "immutable_n": "blue", "mutable": "red"}
        for cls in all_num_objects[m].keys():
            # if cls != 'mutable' and cls != 'immutable_n':
            x1 = all_num_objects[m][cls]
            y1 = all_confidences[m][cls]
            ax1.scatter(x1, y1, color=colors[cls], label=cls, alpha=0.2)
        plt.legend()
        plt.ylabel(f"{m} Confidence")
        plt.xlabel("Number of objects")
        plt.show()

    immutable_dataset = build_dataset("data/immutable_with_aliases.json")  # Homemade
    predictions_imm = load_predictions(f"data/predictions-imm-{model}.json")  # Homemade

    dataset = build_dataset(
        "data/templama/{}_with_aliases.json".format(split)
    )  # Mutable
    predictions_mu = load_predictions(
        f"data/predictions-{split}-{model}.json"
    )  # Mutable

    # relations = build_relations(dataset)
    for relation_id, relation in relations.items():
        print(relation_id, relation.mutation_mean, relation.mutation_std)

    # compute changes
    ratios = list()
    mutables = {
        "imm": defaultdict(list),
        "never": defaultdict(list),
        "rarely": defaultdict(list),
        "often": defaultdict(list),
    }

    # Fetch mutation rate (0.0) and confidence for immutable dataset
    for query in immutable_dataset:
        prediction = get_prediction(predictions_imm, query.id)
        if prediction:
            confidences = sorted(
                [p["first_token_probability"] for p in prediction["predictions"]],
                reverse=True,
            )
            # confidences = sorted([p['perplexity'] for p in prediction['predictions']], reverse=False)
            confidences = [c for c in confidences if not np.isnan(c)]
            confidence = confidences[0]
            ratio = 0.0
            mutables["imm"]["ratios"].append(ratio)
            mutables["imm"]["confidences"].append(confidence)
            mutables["imm"]["average"].append(np.mean(confidences))

    for query in dataset:
        ratio = query.get_ratio()
        relation = relations[query.relation_id]
        relation_ratio = relation.sample_mutation_rate()
        if ratio <= 0:
            ratio += 0.5 * relation_ratio
        prediction = get_prediction(predictions_mu, query.id)
        confidences = sorted(
            [p["first_token_probability"] for p in prediction["predictions"]],
            reverse=True,
        )
        # confidences = sorted([p['perplexity'] for p in prediction['predictions']], reverse=False)
        confidences = [c for c in confidences if not np.isnan(c)]
        confidence = confidences[0]
        if ratio < 0.3:
            mutables["never"]["ratios"].append(ratio)
            mutables["never"]["confidences"].append(confidence)
            mutables["never"]["average"].append(np.mean(confidences))
        elif ratio < 0.6:
            mutables["rarely"]["ratios"].append(ratio)
            mutables["rarely"]["confidences"].append(confidence)
            mutables["rarely"]["average"].append(np.mean(confidences))
        elif ratio >= 0.6:
            mutables["often"]["ratios"].append(ratio)
            mutables["often"]["confidences"].append(confidence)
            mutables["often"]["average"].append(np.mean(confidences))

    all_confidences = list()
    all_ratios = list()
    for mutation, data in mutables.items():
        print(mutation)
        ratios = data["ratios"]
        confidences = data["confidences"]
        averages = data["average"]
        all_confidences += confidences
        all_ratios += ratios
        print("Data points", len(averages))
        print("Average of averages (std)", np.mean(averages), np.std(averages))
        print("Average (std)", np.mean(confidences), np.std(confidences))
        print("Max", np.max(confidences))
        print("Min", np.min(confidences))
    corr = pearsonr(all_confidences, all_ratios)
    print("Correlation", corr)
    plt.scatter(all_ratios, all_confidences)
    plt.ylabel("Confidence")
    plt.xlabel("Mutation rate")
    plt.show()


if __name__ == "__main__":
    main()
