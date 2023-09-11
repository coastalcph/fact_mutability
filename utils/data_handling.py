import json

from mutability.domain import Queries, Query, Answer
from datasets import load_dataset


def build_dataset(data_paths, dataset=None, split=None):
    queries = dict()
    queries_obj = Queries()

    if dataset:
        dataset_split = load_dataset(dataset)[split]
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
            qcode_split = qcode.split("_")
            if len(qcode_split) > 2:
                print(
                    "The prediction contains 3 elements in the qcode ({}), using"
                    " only the first: {}_{}".format(
                        qcode, qcode_split[0], qcode_split[-1]
                    )
                )
                qcode = "{}_{}".join(qcode_split[0], qcode_split[-1])
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
        print("Warning: {} not in predictions".format(qcode))
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
