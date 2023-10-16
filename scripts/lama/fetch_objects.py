import sys
import os
import json
from wikidata.client import Client
from multiprocessing import Pool
from tqdm import tqdm


def process_file(filename):
    client = Client()
    rel_id = filename.split('.')[0]
    data = json.load(open(os.path.join('./data/wikidata/relations_by_freq', filename)))
    results = list()
    for subj_id in tqdm(data, file=sys.stdout):
        subj = client.get(subj_id, load=True)
        if rel_id in subj.attributes['claims']:
            relation = client.get(rel_id)
            objects = list()
            for instance in subj.attributes['claims'][rel_id]:
                try:
                    obj_id = instance['mainsnak']['datavalue']['value']['id']
                    obj = client.get(obj_id, load=True)
                    obj_label = str(obj.label)
                    objects.append({
                        "qid": obj_id,
                        "label": obj_label
                    })
                except:
                    pass
            results.append({
                "qid": subj_id,
                "label": str(subj.label),
                "rel_id": rel_id,
                "relation": str(relation.label),
                "objects": objects
            })
    json.dump(results, open(f"./data/wikidata/objects_by_freq/{rel_id}.json", 'w'), indent=True)


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