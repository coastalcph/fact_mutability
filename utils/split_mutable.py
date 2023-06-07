from collections import defaultdict
import json

from mutability.domain import Queries, Query, Answer


def build_dataset(data_path):
    queries = dict()
    queries_obj = Queries()
    for line in open(data_path):
        data = json.loads(line)
        query_id = "_".join(data['id'].split("_")[:2])
        query = data['query']
        year = data['date']
        if query_id not in queries:
            queries[query_id] = {
                "query": query,
                "answers": list()
            }
        for answer in data['answer']:
            queries[query_id]['answers'].append((answer['name'], year))
    
    for query_id, data in queries.items():
        query = data['query']
        answers = data['answers']
        answers_obj = [Answer(a, y) for a, y in answers]
        query_obj = Query(query_id, query, answers_obj)
        queries_obj.add_query(query_obj)
    
    return queries_obj


def main():
    for split in ['train', 'val']:
        dataset = build_dataset('data/{}_with_aliases.json'.format(split))
        
        # compute changes
        ratios = list()
        mutables = {
            "never": list(),
            "rarely": list(),
            "often": list()
        }
        for query in dataset:
            ratio = query.get_ratio()
            if ratio < 0.05:
                mutables['never'].append(query.dump())
            elif ratio < 0.4:
                mutables['rarely'].append(query.dump())
            elif ratio >= 0.4:
                mutables['often'].append(query.dump())
            ratios.append(ratio)
        print("Never", len(mutables['never']))
        print("Rarely", len(mutables['rarely']))
        print("Often", len(mutables['often']))
        json.dump(mutables, open(f"./data/mutable_{split}.json", "w"), indent=True)
        dataset.plot_ratios()


if __name__ == '__main__':
    main()