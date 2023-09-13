import json

from mutability.domain import Queries, Query, Answer
from datasets import load_dataset


def build_dataset(data_paths, dataset=None, split=None):
    queries = dict()
    queries_obj = Queries()

    # We can use the probing classifier dataset to get the LAMA groundtruth.
    if dataset:
        dataset_split = load_dataset(dataset, use_auth_token=True)[split]
        for data in dataset_split:
            for sub_uri in data["sub_uri"]:
                query_id = "{}_{}".format(sub_uri, data["relation"])
                query = data["template"]
                year = data.get("date", "2021")
                relation = data["relation"]
                if query_id not in queries:
                    queries[query_id] = {
                        "query": query,
                        "relation": relation,
                        "answers": list(),
                    }
                for answer in data["answers"]:
                    queries[query_id]["answers"].append((answer, year, "no_obj_uri"))

    # Note that if the data_paths contain some of the data in the dataset we
    # will override it and use the groundtruth from the datapaths instead.
    for data_path in data_paths:
        for line in open(data_path):
            data = json.loads(line)
            query_id = "_".join(data["id"].split("_")[:2])
            query = data["query"]
            year = data.get("date", "2021")
            relation = data["relation"]
            if query_id not in queries:
                queries[query_id] = {
                    "query": query,
                    "relation": relation,
                    "answers": list(),
                }
            for answer in data["answer"]:
                queries[query_id]["answers"].append(
                    (answer["name"], year, answer["wikidata_id"])
                )

    for query_id, data in queries.items():
        query = data["query"]
        relation = data["relation"]
        answers = data["answers"]
        answers_obj = [Answer(a, y, q) for a, y, q in answers]
        query_obj = Query(query_id, query, answers_obj, relation)
        queries_obj.add_query(query_obj)

    return queries_obj


def load_predictions(data_path):
    predictions = {}
    with open(data_path) as fhandle:
        for line in fhandle:
            data = json.loads(line)
            qcode = data["qcode"]
            # Some of the TempLAMA examples contain more than one subject qcode
            # even though the subjects are written the same and therefore the
            # query is the same. When preprocessing the data we added a '-' to
            # include both qcodes. But we should only count one in the evaluation.
            qcode_split = qcode.split("-")
            if len(qcode_split) > 1:
                print(
                    "The prediction contains 3 elements in the qcode ({}), using"
                    " only the first: {}".format(qcode, qcode_split[-1])
                )
                qcode = qcode_split[-1]
            del data["qcode"]
            non_empty_predictions = []
            for p in data["predictions"]:
                if len(p["answer"]):
                    non_empty_predictions.append(p)
            data["predictions"] = non_empty_predictions
            predictions[qcode] = data

    return predictions


def get_prediction(predictions, qcode, mode=None):
    if qcode not in predictions:
        return {"answer": ""}
    if not len(predictions[qcode]["predictions"]):
        return {"answer": ""}

    if mode is None:
        return predictions[qcode]
    elif mode == "perplexity":
        return sorted(predictions[qcode]["predictions"], key=lambda x: x["perplexity"])[
            0
        ]
    elif mode == "first_token_probability":
        return sorted(
            predictions[qcode]["predictions"],
            key=lambda x: x["first_token_probability"],
        )[-1]
