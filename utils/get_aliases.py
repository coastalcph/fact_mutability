# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from tqdm import tqdm

endpoint_url = "https://query.wikidata.org/sparql"

query1 = """SELECT ?altLabel
{
 VALUES (?wd) {(wd:"""
query2 = """)}
 ?wd skos:altLabel ?altLabel .
 FILTER (lang(?altLabel) = "en")
}"""
query = query1+"Q483020"+query2

def get_aliases(endpoint_url, qcode):
    query = query1 + qcode + query2
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [result['altLabel']['value'] for result in results["results"]["bindings"]]

for split in ['train', 'val', 'test']:
    lines = []
    print(split)
    for line in tqdm(open('data/{}.json'.format(split))):
        line = json.loads(line)
        for i in range(len(line['answer'])):
            qcode = line['answer'][i]['wikidata_id']
            aliases = get_aliases(endpoint_url, qcode)
            line['answer'][i]['name'] = list(set([line['answer'][i]['name']] + aliases))
        lines.append(json.dumps(line))
    
    with open('data/{}_with_aliases.json'.format(split), 'w') as outfile:
        outfile.write('\n'.join(lines))
