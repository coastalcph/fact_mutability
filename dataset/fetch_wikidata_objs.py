import argparse
import json
import logging
import os

import numpy as np
import requests
from tqdm import tqdm

URL_OBJS_FROM_RELATION = "https://www.wikidata.org/w/api.php?action=query&list=backlinks&blnamespace=0&format=json&bllimit={}&&bltitle=Property:{}"
SEED = 7


def get_objs(relation, rng, objs_count=1500, total_objs=3000):
    objects = []
    continue_data = None
    for i in range(0, total_objs, 500):
        try:
            url = URL_OBJS_FROM_RELATION.format(total_objs, relation)
            if i > 0:
                url += f"&blcontinue={continue_data['blcontinue']}"
            answer = requests.get(url)
        except Exception as e:
            logging.warning(
                "Failed to fetch objects for relation {} using url={}".format(
                    relation, URL_OBJS_FROM_RELATION.format(total_objs, relation)
                )
            )
            logging.warning("Ignored exception: {}".format(e))
        answer = json.loads(answer.content)
        objects.extend([item["title"] for item in answer["query"]["backlinks"]])
        if "continue" not in answer:
            logging.warning(
                "There are no more objects available for relation={}".format(relation)
            )
            objs_count = min(objs_count, len(objects))
            break
        continue_data = answer["continue"]
    logging.info(
        "Relation {}, total objects fetched {}".format(relation, len(set(objects)))
    )
    return list(rng.choice(objects, objs_count, replace=False))


def main(args):
    rng = np.random.default_rng(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    for relation in tqdm(args.relations, desc="Relations"):
        objs = get_objs(relation, rng)
        with open(os.path.join(args.output_dir, relation + ".json"), "w") as f:
            json.dump(objs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relations", nargs="+", required=True, help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)
