import numpy as np
import json
from collections import defaultdict
from wikidata.client import Client


def main():
    client = Client()
    data = json.load(open('./data/lama_with_dates.json'))
    queries = defaultdict(list)
    relations_mutation_rates = defaultdict(list)
    for d in data:
        subj = d['subj_id']
        rel = d['relation']
        # if 'start_time' not in d:  # this query has no mutation
        #     relations_mutation_rates[rel].append(0)
        key = f"{subj}-{rel}"
        queries[key].append(d)

    filtered = [d for d in data if 'start_time' in d]
    print(len(data))
    print(len(queries))
    print(len(filtered))

    mutations = defaultdict(list)
    for i in filtered:
        subj = i['subj_id']
        obj = i['obj_id']
        rel = i['relation']
        key = f"{subj}-{rel}"
        mutations[key].append(i)
    print(len(mutations))

    for k, v in mutations.items():
        years = set()
        grouped_by_dates = defaultdict(set)
        current_answer = set()
        changes = 0
        for elem in v:
            start_time = elem['start_time']
            obj = elem['obj']
            grouped_by_dates[start_time].add(obj)
        for start, objs in grouped_by_dates.items():
            years.add(start)
            if len(current_answer):  # check if the answer changed
                if objs != current_answer:  # something changed
                    changes += 1 
            current_answer = objs
        ratio = (changes) / len(years)
        relation = [a['relation'] for a in v][0]
        relations_mutation_rates[relation].append(ratio)
    
    for relation, rates in relations_mutation_rates.items():
        rel = client.get(relation)
        print(rel.label, ";", "{:.2f}".format(np.mean(rates)), ";", len(rates))




if __name__ == '__main__':
    main()