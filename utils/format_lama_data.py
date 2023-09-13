import json
import os

DATA_PATH = 'data/lama/TREx' 
templates = open('data/lama/relations.jsonl', 'r')

queries = []
for line in templates:
    line = json.loads(line)
    relation = line['relation']
    template = line['template']
    if template.endswith('[Y] .'):
        template = template[:-2]
    else:
        continue
    if relation in ['P39', 'P108']:
        continue
    try:
        f = open(os.path.join(DATA_PATH, '{}.jsonl'.format(relation)))
    except:
        print(relation)
        continue
    for line in f:
        line = json.loads(line)
        X = line['sub_label']
        if template.startswith('[X]'):
            X = X.capitalize()
        Y = line['obj_label']
        query = template.replace('[X]', X)
        query = query.replace('[Y]', '_X_.')

        answer = [{'wikidata_id': line['obj_uri'],
                        'name': Y}]
        _id = '{}_{}_{}'.format(line['sub_uri'], relation, '2021')
        queries.append(json.dumps({'query': query, 'answer': answer, 'id': _id, 'relation': relation, 'date': 2021}))

            
with open('data/lama/data.json', 'w') as f:
    f.write('\n'.join(queries))
