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

start_time = 'P580'
end_time = 'P582'

DESIRED_SUBCLASSES = {'architectural structure', 'agreement', 'album', 'animated series', 'application',
                      'architectural structure', 'armed organization', 'art', 'artwork', 'audiovisual work',
                      'automobile model', 'award', 'building', 'business', 'chemical substance', 'clothing', 'collectible',
                      'comics', 'company', 'competition', 'conflict', 'convention', 'country', 'creative work', 'creator',
                      'cultural heritage', 'dance', 'economy', 'educational organization', 'election', 'entity in event',
                      'equipment', 'era', 'ethnic group', 'event', 'facility', 'fictional character', 'fictional entity',
                      'fictional object', 'fictional vehicle', 'film', 'food', 'game', 'geographic entity', 'geographic feature',
                      'geographic region', 'heritage site', 'historical event', 'human', 'human settlement', 'idiom',
                      'infrastructure', 'institute', 'intellectual work', 'island', 'landform', 'language', 'literary work',
                      'military base', 'military operation', 'military unit', 'monument', 'museum', 'music', 'musical ensemble',
                      'musical group', 'musical work', 'natural geographic object', 'natural geographical entity', 'natural park',
                      'news media', 'newspaper', 'nonprofit organization', 'organization', 'painting', 'park','performance work',
                      'person', 'physical good', 'play', 'political movement', 'political party', 'series', 'social phenomenon',
                      'social system', 'software', 'software company', 'song', 'spatial entity', 'sport organization',
                      'sports club', 'sports organization', 'sports team', 'symbol', 'technology', 'television program',
                      'television series', 'television station', 'territorial entity', 'vehicle','video game',
                      'video game character', 'visual artwork', 'voice actor', 'work of art', 'written work'}


def get_superclasses(results):

    """
    Parses the list of all the results and organizes it in a list.
    """

    superclasses = list()
    for i in range(len(results['results']['bindings'])):
      curr_results = results['results']['bindings'][i]
      if curr_results['classLabel']['value'] not in superclasses:
        superclasses.append(curr_results['classLabel']['value'])
      if curr_results['superclassLabel']['value'] not in superclasses:
        superclasses.append(curr_results['superclassLabel']['value'])
    return superclasses


def get_entity_superclasses(entity: str):

    """
    Given an entity string, returns a list of all the superclasses of which this entity is an instance of
    """
    url= 'https://query.wikidata.org/sparql'
    curr_query = """
      SELECT ?class ?classLabel ?superclass ?superclassLabel
      WHERE 
      {
      wd:%s wdt:P279* ?class.
      ?class wdt:P279 ?superclass.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
      } 
    """ % (entity)
    header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36'}
    res = requests.get(url, params={'format': 'json', 'query': curr_query}, headers=header)
    results = res.json()
    print(results)
    supers = get_superclasses(results)
    return list(DESIRED_SUBCLASSES.intersection(set(supers)))


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


def query_wikidata(query):
    client = Client()
    examples = list()
    subj_id = query['qid']
    relation = query['rel_id']
    try:
        subj = client.get(subj_id, load=True)
        classes = get_entity_superclasses(str(subj.label))
        import pdb; pdb.set_trace()
        if relation in subj.attributes['claims']:
            for instance in subj.attributes['claims'][relation]:
                obj_id = instance['mainsnak']['datavalue']['value']['id']
                obj = client.get(obj_id, load=True)
                if 'qualifiers' in instance:
                    if start_time in instance['qualifiers']:
                        start = instance['qualifiers'][start_time][0]['datavalue']['value']['time']
                        if end_time in instance['qualifiers']:
                            end = instance['qualifiers'][end_time][0]['datavalue']['value']['time']
                        else:
                            end = None
                        example = {
                            "subj_id": subj_id,
                            "subj": str(subj.label),
                            "relation": relation,
                            "obj_id": obj_id,
                            "obj": str(obj.label),
                            "start_time": start,
                            "end_time": end
                        }
                        examples.append(example)
                    else:
                        example = {
                            "subj_id": subj_id,
                            "subj": str(subj.label),
                            "relation": relation,
                            "obj_id": obj_id,
                            "obj": str(obj.label),
                        }
                        examples.append(example)
                        # print("No time qualifiers, no mutability")
                else:
                    example = {
                        "subj_id": subj_id,
                        "subj": str(subj.label),
                        "relation": relation,
                        "obj_id": obj_id,
                        "obj": str(obj.label),
                    }
                    examples.append(example)
                    # print("No qualifiers")
        else:
            example = {
                "subj_id": subj_id,
                "subj": str(subj.label),
                "relation": relation
            }
            # print(f"No claim of {subj.label} for rel {relation}")
            examples.append(example)
    except RuntimeError as e:
        print(e)
    except KeyError as e:
        print(e)
    return examples

def main():
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