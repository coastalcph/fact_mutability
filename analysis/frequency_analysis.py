import json
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from dataset import relations_by_type, relations
import matplotlib.pyplot as plt

def main():
    frequencies = dict()
    for t, rels  in relations_by_type.items():
        for r in rels:
            counts = json.load(open(f"./data/wikidata/wikidata_objs_freq/{r}_with_counts.json"))
            for pid, count in counts:
                frequencies[pid] = count
    all_counts = list(frequencies.values())
    bins = np.percentile(all_counts, np.arange(10, 101, 10))
    print(bins)

    models = [
        "alpaca-7b",
        "falcon-7b",
        "falcon-instruct-7b",
        "llama2-7b",
        "llama2-chat-7b",
        "llama-7b",
    ]
    for m in models:
        print(m)
        accs = defaultdict(list)
        counts = defaultdict(list)
        ds = load_dataset(f"coastalcph/fm-updates-{m}")
        predictions = list()
        with open(f"./data/prompt_updates/{m}/predictions.json") as fhandle:
            for l in fhandle:
                line = json.loads(l)
                predictions.append(line)
        for e, p in zip(ds['test'], predictions):
            qid = e['query']['qid']
            frequency = frequencies[qid]
            mutability = relations[e['query']['rel_id']]
            new_target = p['new_target']
            answer = p['predictions'][0]['answer']
            if answer.startswith(new_target):
                acc = 1
            else:
                acc = 0
            accs[mutability].append(acc)        
            counts[mutability].append(frequency)        
        for mut, acc in accs.items():
            acc = np.array(acc)
            print(mut)
            digitized = np.digitize(counts[mut], bins, right=True)
            bin_means = [acc[digitized == i].mean() for i in range(0, len(bins))]
            bin_std = [acc[digitized == i].std() for i in range(0, len(bins))]
            bin_counts = [len(acc[digitized == i]) for i in range(0, len(bins))]
            for m, s, c in zip(bin_means, bin_std, bin_counts):
                print(m, s, c)
        print()
        print()

if __name__ == '__main__':
    main()