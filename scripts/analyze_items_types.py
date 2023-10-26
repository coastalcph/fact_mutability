import json
from collections import defaultdict
import os

def get_entities_types():
    data_path = './data/wikidata/types'
    files = os.listdir(data_path)
    entities = defaultdict(set)
    for f in files:
        data = json.load(open(os.path.join(data_path, f)))
        for qid, value in data:
            entities[qid].add(value)
    for e, t in entities.items():
        entities[e] = list(t)
    json.dump(entities, open('./data/wikidata/types.json', 'w'), indent=True)
    return entities

def queries_coverage_by_type(queries, entity_types, type):
    remainder = list()
    for q in queries:
        types = q['types']
        if type not in types:
            remainder.append(q)
    return remainder


def compute_subject_counts(queries, entities_types):
    ss = defaultdict(int)
    for query in queries:
        subj_type = entities_types[query['qid']]
        for s in subj_type:
            ss[s] += 1
    return ss

def compute_objects_counts(queries, entities_types):
    ss = defaultdict(int)
    for query in queries:
        for obj in query['objects']:
            obj_type = entities_types[obj['qid']]
            for s in obj_type:
                ss[s] += 1
    return ss

def compute_patterns_counts(queries, entities_types):
    ss = defaultdict(int)
    for query in queries:
        subj_type = entities_types[query['qid']]
        query_types = set()
        for s in subj_type:
            for obj in query['objects']:
                obj_type = entities_types[obj['qid']]
                for o in obj_type:
                    query_types.add(f"{s}-{o}")
        for q in query_types:
            ss[q] += 1
    return ss


def main(rel):
    templates = json.load(open('./templates/clean.json'))
    entities_types = get_entities_types()
    data_path = "./data/wikidata/objects_by_freq/"
    # files = os.listdir(data_path)
    print(rel)
    patterns = defaultdict(int)
    ss = defaultdict(int)
    obs = defaultdict(int) 
    subjects = json.load(open(os.path.join(data_path, rel + ".json")))
    print(len(subjects))
    for query in subjects:
        query_types = set()
        subj_type = entities_types[query['qid']]
        for s in subj_type:
            ss[s] += 1
            for obj in query['objects']:
                obj_type = entities_types[obj['qid']]
                for o in obj_type:
                    obs[o] += 1
                    patterns[f"{s}-{o}"] += 1
                    query_types.add(f"{s}-{o}")
        query['types'] = query_types

    remainder = list(subjects)
    covering_subjects = list()
    current_ratio = 0.0
    while len(remainder):
        ss = compute_patterns_counts(remainder, entities_types)
        if len(ss) == 0:
            break
        l = len(remainder)
        sorted_ss = sorted(ss.items(), key=lambda x: x[1], reverse=True)
        p, n = sorted_ss[0]
        remainder = queries_coverage_by_type(remainder, entities_types, p)
        new_l = len(remainder)
        assert l - new_l == n
        if l != new_l:
            ratio = n / len(subjects)
            current_ratio += ratio
            covering_subjects.append((p, current_ratio))
    if rel in templates:
        for t in templates[rel]['templates']:
            print(t)
    else:
        print(f"NO TEMPLATES FOR {rel}")
    for t, p in covering_subjects[:10]:
        print(t, ";", round(p, 2))
    print()

        # remainder = list(subjects)
        # covering_subjects = list()
        # while len(remainder):
        #     ss = compute_objects_counts(remainder, entities_types)
        #     if len(ss) == 0:
        #         break
        #     l = len(remainder)
        #     sorted_ss = sorted(ss.items(), key=lambda x: x[1], reverse=True)
        #     p, n = sorted_ss[0]
        #     remainder = queries_coverage_by_type(remainder, entities_types, p)
        #     new_l = len(remainder)
        #     if l != new_l:
        #         covering_subjects.append((p, n))
        # print(covering_subjects)
        # import pdb; pdb.set_trace()

        # remainder = list(subjects)
        # remainder = [entities_types[obj['qid']] for query in subjects for obj in query['objects']]
        # covering_subjects = list()
        # for p, n in sorted(obs.items(), key=lambda x: x[1], reverse=True)[:5]:
        #     l = len(remainder)
        #     remainder = queries_coverage_by_type(remainder, p)
        #     new_l = len(remainder)
        #     if l != new_l:
        #         covering_subjects.append((p, n))
        #     if not len(remainder):
        #         break
        # print(covering_subjects)
        # print()



if __name__ == '__main__':
    yellows = [
        "P47",
        "P136",
        "P166",
        "P69",
        "P47",
        "P101",
        "P530",
        "P27",
        "P1412",
        "P1303",
        "P190",
    ]
    reds = [
        "P937",
        "P108",
        "P488",
        "P286",
        "P6",
        "P39",
        "P54",
        "P264",
        "P551",
        "P451",
        "P1308",
        "P210",
        "P1037",
    ]
    for p in yellows:
        main(p)
    for r in reds:
        main(r)