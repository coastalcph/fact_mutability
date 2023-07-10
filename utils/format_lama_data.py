import json
import os

DATA_PATH = 'data/lama/TREx' 

for f_name in os.listdir(DATA_PATH):
    if 'P39' in f_name or 'P108' in f_name:
        continue
    queries = []
    with open(os.path.join(DATA_PATH, f_name)) as f:
        for line in f:
            line = json.loads(line)
            relation = line["predicate_id"]
            for evidence in line['evidences']:
                query = evidence["masked_sentence"]
                if not query.endswith('[MASK].'):
                    continue
                query = query.replace('[MASK]', '_X_')
                answer = [{'wikidata_id': line['obj_uri'],
                           'name': evidence['obj_surface']}]
                _id = '{}_{}_{}'.format(line['sub_uri'], relation, '2021')
                queries.append(json.dumps({'query': query, 'answer': answer, 'id': _id, 'relation': relation}))

            
with open('data/lama/data.jsonl', 'w') as f:
    f.write('\n'.join(queries))
