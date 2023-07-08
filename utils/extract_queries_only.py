import json

def format_query(query):
    if query.endswith(' _X_.'):
        return line['query'].replace(' _X_.', '')
    elif query.startswith('_X_ is '):
        query = query.replace('_X_ is ', '')
        query = query[0].upper() + query[1:-1]
        return query + ' is'
    else:
        print(query)

for split in ['templama/train', 'templama/val', 'templama/test', 'immutable/data', 'lama/data']:
    queries = []
    seen_qcodes = []
    for line in open('data/{}.json'.format(split)):
        line = json.loads(line)
        query = format_query(line['query'])
        qcode = '_'.join(line['id'].split('_')[:-1])
        if qcode not in seen_qcodes:
            seen_qcodes.append(qcode)
            queries.append('{}\t{}'.format(qcode, query))
    with open('data/{}.txt'.format(split), 'w') as outfile:
        outfile.write('\n'.join(queries))
