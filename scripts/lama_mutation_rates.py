from wikidata.client import Client
from tqdm import tqdm
from tqdm import tqdm
from multiprocessing import Pool

from utils.data_handling import *

start_time = 'P580'
end_time = 'P582'

def query_wikidata(query):
    client = Client()
    examples = list()
    subj_id, relation = query.id.split("_") 
    print(subj_id, relation)
    subj = client.get(subj_id, load=True)
    if relation in subj.attributes['claims']:
        for instance in subj.attributes['claims'][relation]:
            obj_id = instance['mainsnak']['datavalue']['value']['id']
            obj = client.get(obj_id, load=True)
            if 'qualifiers' in instance:
                if start_time in instance['qualifiers'] and end_time in instance['qualifiers']:
                    start = instance['qualifiers'][start_time][0]['datavalue']['value']['time']
                    end = instance['qualifiers'][end_time][0]['datavalue']['value']['time']
                    example = {
                        "subj_id": subj_id,
                        "subj": subj.label,
                        "relation": relation,
                        "obj_id": obj_id,
                        "obj": obj.label,
                        "start_time": start,
                        "end_time": end
                    }
                    print("hey")
                    examples.append(example)
                else:
                    example = {
                        "subj_id": subj_id,
                        "subj": subj.label,
                        "relation": relation,
                        "obj_id": obj_id,
                        "obj": obj.label,
                    }
                    examples.append(example)
                    # print("No time qualifiers, no mutability")
            else:
                example = {
                    "subj_id": subj_id,
                    "subj": subj.label,
                    "relation": relation,
                    "obj_id": obj_id,
                    "obj": obj.label,
                }
                examples.append(example)
                # print("No qualifiers")
    else:
        example = {
            "subj_id": subj_id,
            "subj": subj.label,
            "relation": relation
        }
        # print(f"No claim of {subj.label} for rel {relation}")
        examples.append(example)
    return examples

def main():
    data = list()
    client = Client()
    immutable_dataset = build_dataset('data/lama.json')  # LAMA
    print(len(immutable_dataset))
    with Pool(24) as p:
        res = tqdm(p.map(query_wikidata, immutable_dataset), total=len(immutable_dataset))
    data = list(map(sum, res))

    # for query in tqdm(immutable_dataset):
    #     subj_id, relation = query.id.split("_") 
    #     print(subj_id, relation)
    #     subj = client.get(subj_id, load=True)
    #     if relation in subj.attributes['claims']:
    #         for instance in subj.attributes['claims'][relation]:
    #             obj_id = instance['mainsnak']['datavalue']['value']['id']
    #             obj = client.get(obj_id, load=True)
    #             if 'qualifiers' in instance:
    #                 if start_time in instance['qualifiers'] and end_time in instance['qualifiers']:
    #                     start = instance['qualifiers'][start_time][0]['datavalue']['value']['time']
    #                     end = instance['qualifiers'][end_time][0]['datavalue']['value']['time']
    #                     example = {
    #                         "subj_id": subj_id,
    #                         "subj": subj.label,
    #                         "relation": relation,
    #                         "obj_id": obj_id,
    #                         "obj": obj.label,
    #                         "start_time": start,
    #                         "end_time": end
    #                     }
    #                     data.append(example)
    #                 else:
    #                     example = {
    #                         "subj_id": subj_id,
    #                         "subj": subj.label,
    #                         "relation": relation,
    #                         "obj_id": obj_id,
    #                         "obj": obj.label,
    #                     }
    #                     print("No time qualifiers, no mutability")
    #             else:
    #                 example = {
    #                     "subj_id": subj_id,
    #                     "subj": subj.label,
    #                     "relation": relation,
    #                     "obj_id": obj_id,
    #                     "obj": obj.label,
    #                 }
    #                 print("No qualifiers")
    #     else:
    #         example = {
    #             "subj_id": subj_id,
    #             "subj": subj.label,
    #             "relation": relation
    #         }
    #         print(f"No claim of {subj.label} for rel {relation}")
    #     data.append(example)

    json.dump(data, open("./data/lama_with_dates.json", "w"))

if __name__ == '__main__':
    main()