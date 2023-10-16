from collections import defaultdict
import json
from scripts.gather_templates import relations

def main():
    queries = defaultdict(list)
    templates = json.load(open('./templates/clean.json'))
    for relation, cls in relations.items():
        t = templates[relation]['templates']
        subjects = json.load(open("./data/wikidata/objects_by_freq/{}.json".format(relation)))
        for subject in subjects:
            subj_label = subject['label']
            subj_id = subject['qid']
            for i, template in enumerate(t):
                query = template.replace("[X]", subj_label)
                query = query.replace("[Y]", "_X_")
                answers = [{"wikidata_id": o['qid'], "name": o['label']} for o in subject['objects']]
                # for object in subject['objects']:
                #     obj_label = object['label']
                #     for template in t:
                #         print(query)
                #         import pdb; pdb.set_trace()
                query_object = {
                    "query": query,
                    "answer": answers,
                    "id": f"{subj_id}_{relation}_{i}",
                    "relation": relation,
                    "date": 2021,
                    "type": cls
                }
                queries[relation].append(query_object)

    for rel, l in queries.items():
        with open(f"./data/queries/{rel}.jsonl", "w") as fhandle:
            for q in l:
                fhandle.write("{}\n".format(json.dumps(q)))
                
if __name__ == '__main__':
    main()