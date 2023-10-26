from collections import defaultdict
import json
import requests
from tqdm import tqdm

from dataset import relations

def find_relation_template(relations, rel_id):
    for relation in relations:
        if relation['relation'] == rel_id:
            return relation


def main():
    templates = defaultdict(dict)

    lama_relations_file = './data/lama/relations.jsonl'
    lama_relations = list()
    with open(lama_relations_file) as fhandle:
        for line in fhandle:
            rel = json.loads(line[:-1])
            lama_relations.append(rel)

    pararel_file = "https://raw.githubusercontent.com/yanaiela/pararel/main/data/pattern_data/graphs_json/{}.jsonl"
    for r in tqdm(relations):
        templates[r]['templates'] = list()

        # lama
        lama_rel = find_relation_template(lama_relations, r)
        if lama_rel:
            templates[r]['templates'].append(lama_rel['template'])
            templates[r]['description'] = lama_rel['description']

        # pararel
        url = pararel_file.format(r)
        res = requests.get(url)
        if res.status_code == 200:
            lines = res.text.split('\n')
            for line in lines:
                if line != '':
                    template = json.loads(line)
                    templates[r]['templates'].append(template['pattern'])
    print(templates)
    json.dump(templates, open('./templates/merged.json', 'w'), indent=True)


if __name__ == '__main__':
    main()