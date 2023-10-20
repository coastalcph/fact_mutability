from collections import defaultdict
import sys
import os
import json
from wikidata.client import Client
from multiprocessing import Pool
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON

def get_objects(wikidata_ids, relation):
    print(len(wikidata_ids))
    url= 'https://query.wikidata.org/sparql'
    query = """
        SELECT ?entity ?entityLabel ?relation ?relationLabel
    WHERE {
    VALUES ?entity {%s}
    ?entity wdt:%s ?relation.
    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en".
    }
    }
    """ % (" ".join(list(wikidata_ids)), relation)
    print(query)
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results['results']['bindings']:
        subj_qid = r['entity']['value'].split("/")[-1]
        subj_label = r['entityLabel']['value']
        obj_qid = r['relation']['value'].split("/")[-1]
        obj_label = r['relationLabel']['value']
        yield subj_qid, subj_label, obj_qid, obj_label


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_file(filename):
    client = Client()
    rel_id = filename.split('.')[0]
    relation = client.get(rel_id)
    data = json.load(open(os.path.join('./data/wikidata/relations_by_freq', filename)))
    data = ["wd:" + d for d in data]
    objects_per_subjects = defaultdict(list)
    for chunk in chunks(list(data), 200):
        res = get_objects(chunk, rel_id)
        for subj_id, subj_label, obj_qid, obj_label in res:
            objects_per_subjects[f"{subj_id}_{subj_label}"].append({"qid": obj_qid, "label": obj_label})

    results = list()
    for subj, objects in objects_per_subjects.items():
        subj_id, subj_label = subj.split("_")
        results.append({
            "qid": subj_id,
            "label": str(subj_label),
            "rel_id": rel_id,
            "relation": str(relation.label),
            "objects": objects
        })
    json.dump(results, open(f"./data/wikidata/objects_by_freq/{rel_id}.json", 'w'), indent=True)

    # for subj_id in tqdm(data, file=sys.stdout):
    #     subj = client.get(subj_id, load=True)
    #     if rel_id in subj.attributes['claims']:
    #         relation = client.get(rel_id)
    #         objects = list()
    #         for instance in subj.attributes['claims'][rel_id]:
    #             try:
    #                 obj_id = instance['mainsnak']['datavalue']['value']['id']
    #                 obj = client.get(obj_id, load=True)
    #                 obj_label = str(obj.label)
    #                 objects.append({
    #                     "qid": obj_id,
    #                     "label": obj_label
    #                 })
    #             except:
    #                 pass
    #         results.append({
    #             "qid": subj_id,
    #             "label": str(subj.label),
    #             "rel_id": rel_id,
    #             "relation": str(relation.label),
    #             "objects": objects
    #         })
    # json.dump(results, open(f"./data/wikidata/objects_by_freq/{rel_id}.json", 'w'), indent=True)


def main():
    files = os.listdir('./data/wikidata/relations_by_freq')
    done = os.listdir('./data/wikidata/objects_by_freq')
    files = [f for f in files if 'json' in f and f not in done]
    print(files)
    with Pool() as p:
        res = tqdm(p.map(process_file, files), total=len(files))
        results = list(res)
    



if __name__ == '__main__':
    main()