import sys
import os
from tqdm import tqdm
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import json

import pageviewapi
import pageviewapi.period

properties = ['P36', 'P189', 'P61',
              'P47', 'P30', 'P78',
              'P112', 'P740', 'P495',
              ]

# restricts some properties to be applied to countries only
constraint = {'P47': 'Q6256', 'P36': 'Q6256', 'P30': 'Q6256', 'P421': 'Q6256', 'P78': 'Q6256'}

endpoint_url = "https://query.wikidata.org/sparql"

query = """SELECT DISTINCT ?cid ?label ?article WHERE {
    <CONSTRAINT>
    ?cid p:<PROPERTY> ?statement0 .
    ?statement0 (ps:<PROPERTY>/(wdt:P279*)) _:anyValue<PROPERTY>.
    OPTIONAL {
        ?cid rdfs:label ?label filter (lang(?label) = "en") .
    }
    OPTIONAL {
      ?article schema:about ?cid .
      ?article schema:inLanguage "en" .
      FILTER (SUBSTR(str(?article), 1, 25) = "https://en.wikipedia.org/")
    }
    }
    LIMIT 10000"""

object_query = """SELECT ?item ?label
WHERE
{
  wd:<SUBJECT> wdt:<PROPERTY> ?item.
  OPTIONAL {
        ?item rdfs:label ?label filter (lang(?label) = "en") .
    }
}"""

def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()

def get_pageviews(label):
    try:
        return pageviewapi.period.avg_last('en.wikipedia', label.replace(' ', '_'), last=3650)
    except:
        return 0

### PHASE 1 - Extract entities from Wikidata

for prop in properties:
    print(prop)
    if os.path.exists('data/enrichment/{}.json'.format(prop)):
        continue

    subjects = []
    temp_query = query.replace('<CONSTRAINT>', '?cid wdt:P31 wd:{} .'.format(constraint[prop])) if prop in constraint else query.replace('<CONSTRAINT>', '')
    temp_query = temp_query.replace('<PROPERTY>', prop)
    
    results = get_results(endpoint_url, temp_query)
    for result in tqdm(results["results"]["bindings"]):
        if len(result["cid"]["value"]):
            qcode = result["cid"]["value"].split('/')[-1]
        else:
            continue
        if "label" in result:
            label = result["label"]["value"]
        else:
            continue
        if "article" in result:
            hits = get_pageviews(result["article"]["value"].split('/')[-1])
        else:
            continue
        subjects.append({'qcode': qcode, 'label':label, 'hits': hits})
    
    top_subjects = sorted(subjects, key=lambda d: d['hits'])[-100:] 
    
    # find objects
    for subject in top_subjects:
        subject['objects'] = []
        
        temp_query = object_query.replace('<SUBJECT>', subject['qcode'])
        temp_query = temp_query.replace('<PROPERTY>', prop)
        results = get_results(endpoint_url, temp_query) 
        for result in results["results"]["bindings"]:
            qcode = result["item"]["value"].split('/')[-1]
            if "label" in result:
                label = result["label"]["value"]
            else:
                continue
            subject['objects'].append({'qcode': qcode, 'label':label})
    
    with open('data/enrichment/{}.json'.format(prop), 'w') as f:
        json.dump(top_subjects, f)

### PHASE 2 - Format entities into templates 

templates = {'P47': '{} shares borders with _X_.',
            'P36': 'The capital of {} is _X_.',
            'P30': '{} is found on the continent of _X_.',
            'P78': 'The top-level internet domain of {} is _X_.',
            'P112': '{} was founded by _X_.',
            'P740': '{} comes from _X_.',
            'P495': '{} originated in the country of _X_.',
            'P61': '{} was discovered or invented by _X_.',
            'P189': 'The place of discovery of {} is _X_.'}

data = []

for prop in properties:
    wikidata = json.load(open('data/enrichment/{}.json'.format(prop)))
    for subj_item in wikidata:
        subj = subj_item['label']
        query = templates[prop].format(subj)
        query = query[0].upper() + query[1:]
        answers = []
        for obj_item in subj_item['objects']:
            answers.append({"wikidata_id": obj_item['qcode'], "name": obj_item['label']})
        data.append({'query': query,
                     'date': 2023,
                     'answer': answers,
                     'id': '{}_{}_{}'.format(subj_item['qcode'], prop, '2023')})

with open('data/immutable.json', 'w') as f:
    f.write('\n'.join([json.dumps(line) for line in data]))
