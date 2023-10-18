import argparse
import collections
import os

from datasets import load_dataset
from tqdm import tqdm


def format_query(query):
    query = query.replace("_X_ .", "_X_.")
    if query.endswith(" _X_."):
        return query.replace(" _X_.", "")
    else:
        raise Exception("The query does not end with '_X_.': '{}'".format(query))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    template_id_to_queries = collections.defaultdict(dict)
    ds = load_dataset("coastalcph/fm_queries")["train"]
    for line in tqdm(ds):
        query = format_query(line["query"])
        # entity-code_relation-code_template-id
        entity_id, relation_id, template_id = line["id"].split("_")
        qcode = f"{entity_id}_{relation_id}"
        if qcode not in template_id_to_queries[template_id]:
            template_id_to_queries[template_id][qcode] = "{}\t{}".format(qcode, query)
        else:
            print(
                "Warning. Repeated subject '{}' for relation '{}'. Ignoring this"
                " instance.".format(entity_id, relation_id)
            )
    for template_id in template_id_to_queries.keys():
        with open(
            os.path.join(args.output_dir, f"fm_queries_{template_id}.txt"), "w"
        ) as outfile:
            outfile.write("\n".join(template_id_to_queries[template_id].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--output_dir",
        default="data/",
        type=str,
        help="",
    )
    args = parser.parse_args()
    main(args)
