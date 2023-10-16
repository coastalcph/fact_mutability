from collections import defaultdict
import os
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

queries_path = './data/queries'
relations = os.listdir(queries_path)
qcodes = set()
for rel_file in relations:
    rel = rel_file.split('.')[0]
    print(rel)
    lines = []
    for line in tqdm(open(os.path.join(queries_path, rel_file))):
        line = json.loads(line)
        for i in range(len(line['answer'])):
            qcode = line['answer'][i]['wikidata_id']
            qcodes.add(qcode)

qcodes_aliases = defaultdict(list)
for qcode in tqdm(qcodes):
    aliases = get_aliases(endpoint_url, qcode)
    qcodes_aliases[qcode] = aliases
    
json.dump(qcodes_aliases, open('data/objects_with_aliases.json', 'w'))
