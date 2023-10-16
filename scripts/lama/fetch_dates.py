from wikidata.client import Client
from tqdm import tqdm
from tqdm import tqdm
from multiprocessing import Pool
from urllib.error import HTTPError

from utils.data_handling import *

start_time = 'P580'
end_time = 'P582'

def query_wikidata(query):
    client = Client()
    examples = list()
    subj_id, relation = query.id.split("_") 
    try:
        subj = client.get(subj_id, load=True)
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
    immutable_dataset = build_dataset('data/lama.json')  # LAMA
    print(len(immutable_dataset))
    for query in tqdm(immutable_dataset, total=len(immutable_dataset)):
        exs = query_wikidata(query)
        data += exs
    json.dump(data, open("./data/lama_with_dates.json", "w"))

if __name__ == '__main__':
    main()