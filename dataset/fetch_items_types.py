import sys
import requests
import os
from wikidata.client import Client
from tqdm import tqdm
from tqdm import tqdm
from multiprocessing import Pool
from urllib.error import HTTPError
from SPARQLWrapper import SPARQLWrapper, JSON

from utils.data_handling import *


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_types(wikidata_ids):
    print(len(wikidata_ids))
    url= 'https://query.wikidata.org/sparql'
    query = """
    SELECT ?entity ?instanceOf ?instanceOfLabel
        WHERE {
        VALUES ?entity {%s} # List of Wikidata entity IDs
        ?entity wdt:P31 ?instanceOf.
        
        SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en".
            ?instanceOf rdfs:label ?instanceOfLabel.
        }
    }
    """ % " ".join(list(wikidata_ids)[:1000])
    print(query)
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results['results']['bindings']:
        qid = r['entity']['value'].split("/")[-1]
        value = r['instanceOfLabel']['value']
        yield qid, value


def main():
    """Fetch the type of items (subject/object) e.g. Barack Obama = human"""
    data = list()
    data_path = './data/wikidata/objects_by_freq/'
    files = os.listdir(data_path)
    for f in files:
        results = list()
        qids = set()
        current_queries = json.load(open(os.path.join(data_path, f)))
        for query in tqdm(current_queries, total=len(current_queries)):
            subj_id = query['qid']
            qids.add(f"wd:{subj_id}")
            for object in query['objects']:
                obj_id = object['qid']
                qids.add(f"wd:{obj_id}")
        for chunk in chunks(list(qids), 200):
            res = get_types(chunk)
            for qid, value in res:
                results.append((qid, value))
        json.dump(results, open(f"./data/wikidata/types/{f}", "w"))

if __name__ == '__main__':
    main()