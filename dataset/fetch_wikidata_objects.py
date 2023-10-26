import argparse
import json
import logging
import os

import numpy as np
import requests
import wandb
from tqdm import tqdm

URL_OBJS_FROM_RELATION = "https://www.wikidata.org/w/api.php?action=query&list=backlinks&blnamespace=0&format=json&bllimit={}&&bltitle=Property:{}"
URL_SITELINKS_FOR_ENTITY = "https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={}&props=sitelinks/urls"
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
                    relation, url
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


def get_obj_sitelink_count(objs):
    objs_and_count = []
    for qcode in objs:
        url = URL_SITELINKS_FOR_ENTITY.format(qcode)
        try:
            answer = requests.get(url)
        except Exception as e:
            logging.warning(
                "Failed to fetch sitelinks for entity {} using url={}".format(
                    qcode, url
                )
            )
            logging.warning("Ignored exception: {}".format(e))
        answer = json.loads(answer.content)
        objs_and_count.append((qcode, len(answer["entities"][qcode]["sitelinks"])))
    return sorted(objs_and_count, key=lambda x: x[1], reverse=True)


def main(args):
    rng = np.random.default_rng(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    for relation in tqdm(args.relations, desc="Relations"):
        objs = get_objs(relation, rng, 4000, 4000)
        objs_and_count = get_obj_sitelink_count(objs)
        with open(
            os.path.join(args.output_dir, relation + "_with_counts.json"), "w"
        ) as f:
            json.dump(objs_and_count, f)
        with open(os.path.join(args.output_dir, relation + ".json"), "w") as f:
            json.dump([i for i, _ in objs_and_count[:1500]], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--relations", nargs="+", required=True, help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    wandb.init(project="fetch_wikidata_objs", config=args)

    main(args)
