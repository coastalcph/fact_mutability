import os
from collections import defaultdict
import json

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from python2latex import Document, Table, bold

from mutability.domain import Relation
from utils.data_handling import *


from dataset import relations, sorted_relations_by_type


np.random.seed(42)

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
    models = ["alpaca", "llama"]
    templates = [0, 1, 2, 3, 4]
    for m in models:
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
    
    col = len(stats_by_model) + 2
    row = len(stats_by_model['alpaca'].keys()) + 2 + len(sorted_relations_by_type) + 1
    doc = Document(filename='f1_scores', filepath='output/tables', doc_type='standalone', border='10pt')
    table = doc.new(Table(shape=(row, col), as_float_env=False, alignment=['c','l','c','c']))
    table[0,2:] = 'Models'
    table[0,2:].apply_command(bold)
    table[1,1] = 'Relation'
    table[1,2:] = [m.capitalize() for m in stats_by_model.keys()]
    table[1,1:].apply_command(bold)
    table[2:,1] = rels_labels
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
        table[2:, i+2] = data

    # Header rule
    table[0,2:].add_rule()
    #  rule
    table[1,1:].add_rule()

    tex = doc.build(compile_to_pdf=False, show_pdf=False)
    print(tex)


    all_num_objects = defaultdict(lambda: defaultdict(list))
    all_confidences = defaultdict(lambda: defaultdict(list))
    models = [("alpaca", "stanford_alpaca"), ("llama", "llama")]
    for m, k in models:
        for relation, cls in relations.items():
            t = template_by_model[m][relation]
            predictions = load_predictions(f"../fm_predictions/fm_queries_v2/{m}_fmv2_final_{t}---projects-nlp-data-constanzam-{k}-huggingface-ckpts-7B/predictions.json")  # LAMA
            subjects = json.load(open("./data/wikidata/objects_by_freq/{}.json".format(relation)))
            for subject in subjects:
                subj_label = subject['label']
                subj_id = subject['qid']
                prediction = predictions[f"{subj_id}_{relation}"]
                confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
                confidences = [c for c in confidences if not np.isnan(c)]
                if len(confidences) and len(subject['objects']):
                    confidence = confidences[0]
                    num_objects = len(subject['objects'])
                    all_confidences[m][cls].append(confidence)
                    all_num_objects[m][cls].append(num_objects)
            # import pdb; pdb.set_trace()

    # corr = pearsonr(all_confidences, all_num_objects)
    # print("Correlation", corr)

    for m in all_num_objects.keys():
        print(m)
        for cls in all_num_objects[m].keys():
            print(cls, np.mean(all_confidences[m][cls]), np.std(all_confidences[m][cls]))
        print()

    for m in all_num_objects.keys():
        fig = plt.figure()
        ax1 = fig.add_subplot()
        colors = {
            'immutable': 'green',
            'immutable_n': 'blue',
            'mutable': 'red'
        }
        for cls in all_num_objects[m].keys():
            # if cls != 'mutable' and cls != 'immutable_n':
            x1 = all_num_objects[m][cls]
            y1 = all_confidences[m][cls]
            ax1.scatter(x1, y1, color=colors[cls], label=cls, alpha=0.2)
        plt.legend()
        plt.ylabel(f"{m} Confidence")
        plt.xlabel("Number of objects")
        plt.show()

    # immutable_dataset = build_dataset('data/lama_with_aliases.json')  # LAMA

    immutable_dataset = build_dataset('data/immutable_with_aliases.json')  # Homemade
    predictions_imm = load_predictions(f"data/predictions-imm-{model}.json")  # Homemade
    
    dataset = build_dataset('data/templama/{}_with_aliases.json'.format(split))  # Mutable
    predictions_mu = load_predictions(f"data/predictions-{split}-{model}.json")  # Mutable

    # relations = build_relations(dataset)
    for relation_id, relation in relations.items():
        print(relation_id, relation.mutation_mean, relation.mutation_std)

    # compute changes
    ratios = list()
    mutables = {
        "imm": defaultdict(list),
        "never": defaultdict(list),
        "rarely": defaultdict(list),
        "often": defaultdict(list)
    }

    # Fetch mutation rate (0.0) and confidence for immutable dataset
    for query in immutable_dataset:
        prediction = get_prediction(predictions_imm, query.id)
        if prediction:
            confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
            # confidences = sorted([p['perplexity'] for p in prediction['predictions']], reverse=False)
            confidences = [c for c in confidences if not np.isnan(c)]
            confidence = confidences[0]
            ratio = 0.0
            mutables['imm']['ratios'].append(ratio)
            mutables['imm']['confidences'].append(confidence)
            mutables['imm']['average'].append(np.mean(confidences))

    for query in dataset:
        ratio = query.get_ratio()
        relation = relations[query.relation_id]
        relation_ratio = relation.sample_mutation_rate()
        if ratio <= 0:
            ratio += 0.5 * relation_ratio
        prediction = get_prediction(predictions_mu, query.id)
        confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
        # confidences = sorted([p['perplexity'] for p in prediction['predictions']], reverse=False)
        confidences = [c for c in confidences if not np.isnan(c)]
        confidence = confidences[0]
        if ratio < 0.3:
            mutables['never']['ratios'].append(ratio)
            mutables['never']['confidences'].append(confidence)
            mutables['never']['average'].append(np.mean(confidences))
        elif ratio < 0.6:
            mutables['rarely']['ratios'].append(ratio)
            mutables['rarely']['confidences'].append(confidence)
            mutables['rarely']['average'].append(np.mean(confidences))
        elif ratio >= 0.6:
            mutables['often']['ratios'].append(ratio)
            mutables['often']['confidences'].append(confidence)
            mutables['often']['average'].append(np.mean(confidences))
    
    all_confidences = list()
    all_ratios = list()
    for mutation, data in mutables.items():
        print(mutation)
        ratios = data['ratios']
        confidences = data['confidences']
        averages = data['average']
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


if __name__ == '__main__':
    main()
