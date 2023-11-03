from datasets import load_dataset
from collections import Counter
import numpy as np
import itertools

SPLIT_TO_RELATIONS = {
    "train": ["P103", "P19", "P937", "P286", "P6", "P159", "P27", "P1412", "P190"],
    "validation": ["P20", "P364", "P108", "P488", "P69", "P101"],
    "test": ["P36", "P449", "P39", "P264", "P47", "P136"],
}


if __name__ == "__main__":
    ds = load_dataset("coastalcph/fm_queries")
    ds["all_fm"] = ds["train"]
    rng = np.random.default_rng(7)
    print("Examples per relation", Counter(ds["all_fm"]["relation"]))
    split_to_subjs = {}
    split_to_objs = {}
    for split in SPLIT_TO_RELATIONS.keys():
        ds[split] = ds["all_fm"].filter(
            lambda ex: ex["relation"] in SPLIT_TO_RELATIONS[split]
        )
        split_to_subjs[split] = set([id_.split("_")[0] for id_ in ds[split]["id"]])
        split_to_objs[split] = set(
            [a["wikidata_id"] for answers in ds[split]["answer"] for a in answers]
        )

    overlapping_subjs = set()
    overlapping_objs = set()
    for split_i, split_j in itertools.combinations(SPLIT_TO_RELATIONS.keys(), 2):
        print(split_i, "-", split_j)
        subjs = split_to_subjs[split_i].intersection(split_to_subjs[split_j])
        objs = split_to_objs[split_i].intersection(split_to_objs[split_j])
        print("overlapping subjects", len(subjs))
        print("overlapping objects", len(objs))
        overlapping_subjs.update(subjs)
        overlapping_objs.update(objs)

    for split in SPLIT_TO_RELATIONS.keys():
        ds[split] = ds[split].filter(
            lambda ex: ex["id"].split("_")[0] not in overlapping_subjs
            and not set(
                [answer["wikidata_id"] for answer in ex["answer"]]
            ).intersection(overlapping_objs)
        )
        # Select one of the 5 templates at random.
        tuple_id_to_index = {
            s: i for i, s in enumerate(list(set([id_[:-2] for id_ in ds[split]["id"]])))
        }
        assert int(len(ds[split]) / 5) == len(tuple_id_to_index)
        template_choice = rng.choice(5, int(len(ds[split]) / 5))
        ds[split] = ds[split].filter(
            lambda ex: template_choice[tuple_id_to_index[ex["id"][:-2]]]
            == int(ex["id"][-1])
        )
    ds = ds.map(lambda ex: {"is_mutable": 1 if ex["type"] == "mutable" else 0})
    ds_1_n = ds.filter(lambda ex: ex["type"] != "immutable")
    ds_1_1 = ds.filter(lambda ex: ex["type"] != "immutable_n")
    print("Dataset mutable (1-1) vs immutable", ds_1_1)
    print("Dataset mutable (1-N) vs immutable", ds_1_n)
    print("Pushing to the hub...")
    ds_1_1.push_to_hub("coastalcph/fm_classifier-1-1")
    ds_1_n.push_to_hub("coastalcph/fm_classifier-1-n")
