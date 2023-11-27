import json
import collections
import os
from glob import glob
import numpy as np
from datasets import load_dataset

RESULTS_DIR = "/projects/nlp/data/constanzam/mdl_mutability/"


def get_acc(dir_name, n=10):
    f = glob(os.path.join(dir_name, f"{n}_*/test_results.json"))
    if len(f) == 0:
        f = glob(os.path.join(dir_name, f"{n}_*/f_test_results.json"))
    assert len(f) == 1, f"{dir_name} {n}"
    with open(f[0]) as test_results:
        data = json.load(test_results)
        key = [k for k in data.keys() if k.endswith("accuracy")]
        assert len(key) == 1, data.keys()
        return data[key[0]]


def get_code_length(dir_name, n):
    f = glob(os.path.join(dir_name, f"{n}_*/online_portion_results.json"))
    if len(f) == 0:
        f = glob(os.path.join(dir_name, f"{n}_*/f_online_portion_results.json"))
    assert len(f) == 1, f"{dir_name} {n}"
    with open(f[0]) as eval_results:
        data = json.load(eval_results)
        key = [k for k in data.keys() if k.endswith("sum_log2_prob")]
        assert len(key) == 1, data.keys()
        return data[key[0]]


def compute_online_codelength(dir_name):
    log2 = collections.defaultdict(dict)
    for i in range(0, 10):
        log2[i] = -get_code_length(dir_name, i)
    print(dir_name)
    print(log2)
    return sum([log2[i] for i in range(0, 10)])


def print_metrics(ds, dir_name):
    num_classes = 2
    first_portion_size = 0.1
    uniform_codelength_first_batch = int(
        first_portion_size * 0.01 * len(ds["train"])
    ) * np.log2(num_classes)
    online_codelength = uniform_codelength_first_batch + compute_online_codelength(
        dir_name
    )
    uniform_codelength = len(ds["train"]) * np.log2(num_classes)
    compression = uniform_codelength / online_codelength
    cl_trained_all_data = get_code_length(dir_name, 10)
    # model_cl + correction = online_cl
    # model_cl = online_cl - correction
    print(
        "labels: codelength={} compression={} model_cl={}".format(
            online_codelength, compression, online_codelength - cl_trained_all_data
        )
    )
    print("Final prob. acc:", get_acc(dir_name))


clf_type = "1-n"
wu = {"1-1": "0.2", "1-n": "0.1"}
model = "llama2-7b"
model_to_results_dir = {
    "llama-7b": "/projects/nlp/data/constanzam/mdl_mutability",
    "llama2-7b": f"/projects/nlp/data/constanzam/mdl_mutability/llama2-7b/fm_dataset_{clf_type}/",
    "alpaca-7b": f"/projects/nlp/data/constanzam/mdl_mutability/alpaca-7b/fm_dataset_{clf_type}/",
}
model_to_normal_subfolder = {
    "llama-7b": "no_overlap_fix_fm_dataset_1-1"
    if clf_type == "1-1"
    else "llama-7B/fm_dataset_1-n/lr5e-5_wu0.2_no_overlap_fix",
    "llama2-7b": f"lr5e-5_wu{wu[clf_type]}_",
    "alpaca-7b": f"lr5e-5_wu{wu[clf_type]}_",
}
model_to_rand_subfolder = {
    "llama-7b": f"no_overlap_fix_rand_fm_dataset_{clf_type}"
    if clf_type == "1-1"
    else "llama-7B/fm_dataset_1-n/lr5e-5_wu0.2_rand",
    "llama2-7b": f"lr5e-5_wu{wu[clf_type]}_rand",
    "alpaca-7b": f"lr5e-5_wu{wu[clf_type]}_rand",
}

ds = load_dataset(f"coastalcph/mutability_classifier-{clf_type}")
print("--- normal ----")
print_metrics(
    ds, os.path.join(model_to_results_dir[model], model_to_normal_subfolder[model])
)
print("--- random ----")
print_metrics(
    ds, os.path.join(model_to_results_dir[model], model_to_rand_subfolder[model])
)
