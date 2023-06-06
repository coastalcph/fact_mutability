import json

for split in ['train', 'val', 'test']:
    queries = []
    for line in open('data/{}.json'.format(split)):
        query = json.loads(line)['query'].replace(' _X_.', '')
        if query not in queries:
            queries.append(query)
    with open('data/{}.txt'.format(split), 'w') as outfile:
        outfile.write('\n'.join(queries))
