import json
import collections
import os
from glob import glob
import numpy as np
from datasets import load_dataset

def compute_online_codelength(dir_name):
    log2 = collections.defaultdict(dict)
    for f in glob(os.path.join(dir_name, '**/eval_results.json')):
        if f.startswith(os.path.join(dir_name, 'fixed/10')):
            continue
        t = f[len(dir_name)]
        with open(f) as eval_results:
            data = json.load(eval_results)
            log2[t] = data['eval_sum_log2_prob']
    return log2['0'] - sum([log2[str(i)] for i in range(1, 10)])

def compute_compression(codelength, ds):
    portion_sizes = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
    portion_indices = [
        int(portion_size * 0.01 * len(ds["train"]))
        for portion_size in portion_sizes
    ]
    num_classes = 2
    uniform_codelength = len(ds["train"]) * np.log2(num_classes)
    return uniform_codelength / codelength

ds = load_dataset("cfierro/mutability_classifier_data", use_auth_token=True)

online_codelength = compute_online_codelength('/projects/nlp/data/constanzam/mdl_mutability/normal/')
compression = compute_compression(online_codelength, ds)
print(f'normal labels: codelength={online_codelength} compression={compression}')

online_codelength = compute_online_codelength('/projects/nlp/data/constanzam/mdl_mutability/random_r/')
compression = compute_compression(online_codelength, ds)
print(f'random labels: codelength={online_codelength} compression={compression}')
