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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_aliases(endpoint_url, wikidata_ids):
    query = """
    SELECT ?wd ?altLabel
    {
    VALUES ?wd {%s}
    ?wd skos:altLabel ?altLabel .
    FILTER (lang(?altLabel) = "en")
    }
    """ % " ".join(list(wikidata_ids)[:1000])
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    aliases_by_qcodes = defaultdict(set)
    for result in results["results"]["bindings"]:
        qcode = result['wd']['value'].split("/")[-1]
        alias = result['altLabel']['value']
        aliases_by_qcodes[qcode].add(alias)
    return aliases_by_qcodes 

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
            qcodes.add(f"wd:{qcode}")

qcodes_aliases = defaultdict(list)
for chunk in tqdm(chunks(list(qcodes), 200), total=len(qcodes)/200):
    aliases = get_aliases(endpoint_url, chunk)
    for qcode, alias in aliases.items(): 
        qcodes_aliases[qcode] = list(alias)
    
json.dump(qcodes_aliases, open('data/objects_with_aliases.json', 'w'))
