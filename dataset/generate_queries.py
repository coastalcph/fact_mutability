from collections import defaultdict
import json
from datasets import load_dataset


from dataset import relations

def main():
    queries = defaultdict(list)
    templates = load_dataset("coastalcph/fm_templates", split="train")
    for relation, cls in relations.items():
        t = templates[relation][0]['templates']
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

    with open(f"./data/queries.jsonl", "w") as fhandle_1:
        for rel, l in queries.items():
            with open(f"./data/queries/{rel}.jsonl", "w") as fhandle_2:
                for q in l:
                    fhandle_1.write("{}\n".format(json.dumps(q)))
                    fhandle_2.write("{}\n".format(json.dumps(q)))
                
if __name__ == '__main__':
    main()