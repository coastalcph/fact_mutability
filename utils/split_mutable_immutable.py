from collections import defaultdict
import json

from mutability.domain import Queries, Query, Answer

def plot_ratios(ratios):
    plt.hist(ratios)
    plt.show()

def build_dataset(data_path):
    queries = defaultdict(list)
    queries_obj = Queries()
    for line in open(data_path):
        data = json.loads(line)
        query = data['query']
        answer = data['answer'][0]['name']
        year = data['date']
        queries[query].append((answer, year))
    
    for query, answers in queries.items():
        answers_obj = [Answer(a, y) for a, y in answers]
        query_obj = Query(query, answers_obj)
        queries_obj.add_query(query_obj)
    
    return queries_obj


def main():
    for split in ['train', 'val', 'test']:
        dataset = build_dataset('data/{}.json'.format(split))
        
        # compute changes
        ratios = list()
        mutables = {
            "never": list(),
            "rarely": list(),
            "often": list()
        }
        for query in dataset:
            ratio = query.get_ratio()
            if ratio < 0.1:
                mutables['never'].append(query.dump())
            elif ratio < 0.5:
                mutables['rarely'].append(query.dump())
            elif ratio >= 0.5:
                mutables['often'].append(query.dump())
            ratios.append(ratio)
        json.dump(mutables, open(f"./data/mutable_{split}.json", "w"), indent=True)
        dataset.plot_ratios()


if __name__ == '__main__':
    main()