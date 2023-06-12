import json

from mutability.domain import Queries, Query, Answer

def build_dataset(data_path):
    queries = dict()
    queries_obj = Queries()
    for line in open(data_path):
        data = json.loads(line)
        query_id = "_".join(data['id'].split("_")[:2])
        query = data['query']
        year = data['date']
        relation = data['relation']
        if query_id not in queries:
            queries[query_id] = {
                "query": query,
                "relation": relation,
                "answers": list()
            }
        for answer in data['answer']:
            queries[query_id]['answers'].append((answer['name'], year, answer['wikidata_id']))

    for query_id, data in queries.items():
        query = data['query']
        relation = data['relation']
        answers = data['answers']
        answers_obj = [Answer(a, y, q) for a, y, q in answers]
        query_obj = Query(query_id, query, answers_obj, relation)
        queries_obj.add_query(query_obj)

    return queries_obj


def load_predictions(data_path):
    predictions = {}
    with open(data_path) as fhandle:
        for line in fhandle:
            data = json.loads(line)
            qcode = data['qcode']
            del data['qcode']
            predictions[qcode] = data
    
    return predictions

def get_prediction(predictions, qcode, mode=None):
    if qcode not in predictions:
        return None

    if mode is None:
        return predictions[qcode]
    elif mode == 'perplexity':
        return sorted(predictions[qcode]['predictions'], key=lambda x: x['perplexity'])[0]
    elif mode == 'first_token_probability':
        return sorted(predictions[qcode]['predictions'], key=lambda x: x['first_token_probability'])[-1]

