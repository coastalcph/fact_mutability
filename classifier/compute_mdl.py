import json
import collections
import os
from glob import glob
import numpy as np
from datasets import load_dataset

RESULTS_DIR = "/projects/nlp/data/constanzam/mdl_mutability/"


def get_acc(dir_name, n=10):
    f = glob(os.path.join(dir_name, f"{n}_*/test_results.json"))
    assert len(f) == 1
    with open(f[0]) as test_results:
        return json.load(test_results)["eval_accuracy"]


def get_code_length(dir_name, n):
    f = glob(os.path.join(dir_name, f"{n}_*/online_portion_results.json"))
    assert len(f) == 1
    with open(f[0]) as eval_results:
        return json.load(eval_results)["eval_sum_log2_prob"]


def compute_online_codelength(dir_name):
    log2 = collections.defaultdict(dict)
    for i in range(0, 10):
        log2[i] = get_code_length(dir_name, i)
    print(dir_name)
    print(log2)
    return log2[0] - sum([log2[i] for i in range(1, 10)])


def compute_compression(codelength, ds):
    num_classes = 2
    uniform_codelength = len(ds["train"]) * np.log2(num_classes)
    return uniform_codelength / codelength


def print_metrics(dir_name):
    online_codelength = compute_online_codelength(dir_name)
    compression = compute_compression(online_codelength, ds)
    cl_trained_all_data = get_code_length(dir_name, 10)
    # model_cl + correction = online_cl
    # model_cl = online_cl - correction
    print(
        "normal labels: codelength={} compression={} model_cl={}".format(
            online_codelength, compression, online_codelength - cl_trained_all_data
        )
    )
    print("Final prob. acc:", get_acc(dir_name))


ds = load_dataset("coastalcph/fm_queries_classifier", use_auth_token=True)
print_metrics(os.path.join(RESULTS_DIR, "normal_fm_dataset/"))
print()
print_metrics(os.path.join(RESULTS_DIR, "random_fm_dataset/"))
