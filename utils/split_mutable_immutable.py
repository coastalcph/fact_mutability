from collections import defaultdict
import json

import matplotlib.pyplot as plt

def plot_ratios(ratios):
    plt.hist(ratios)
    plt.show()

def main():
    for split in ['train', 'val', 'test']:
        queries = defaultdict(list)
        for line in open('data/{}.json'.format(split)):
            data = json.loads(line)
            query = data['query']
            answer = data['answer'][0]['name']
            year = data['date']
            queries[query].append((answer, year))

        # compute changes
        ratios = list()
        mutables = {
            "never": list(),
            "rarely": list(),
            "often": list()
        }
        for query, values in queries.items():
            unique_names = set()
            years = set()
            for name, year in values:
                unique_names.add(name)
                years.add(year)
            ratio = len(unique_names) / len(years)
            if ratio < 0.2:
                mutables['never'].append(query)
            elif ratio < 0.5:
                mutables['rarely'].append(query)
            elif ratio >= 0.5:
                mutables['often'].append(query)
            json.dump(mutables, open(f"./data/mutable_{split}.json", "w"), indent=True)
            ratios.append(ratio)
        # plot_ratios(ratios)


if __name__ == '__main__':
    main()