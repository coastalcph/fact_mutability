from collections import defaultdict
import json
import requests
from tqdm import tqdm

relations = {
    "P740": "immutable",
    "P103": "immutable",
    "P19": "immutable",
    "P20": "immutable",
    "P30": "immutable",
    "P36": "immutable",
    "P159": "immutable",
    "P449": "immutable",
    "P364": "immutable",
    "P495": "immutable",
    "P140": "immutable",
    "P138": "immutable",
    "P47": "immutable_n",
    "P136": "immutable_n",
    "P937": "mutable",
    "P108": "mutable",
    "P488": "mutable",
    "P286": "mutable",
    "P6": "mutable",
    "P39": "mutable",
    "P54": "mutable",
    "P264": "mutable",
    "P551": "mutable",
    "P451": "mutable",
    "P1308": "mutable",
    "P210": "mutable",
    "P1037": "mutable",
}


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
    json.dump(templates, open('./data/templates/merged.json', 'w'), indent=True)


if __name__ == '__main__':
    main()