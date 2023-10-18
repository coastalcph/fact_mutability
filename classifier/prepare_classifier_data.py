from datasets import load_dataset
from collections import Counter
import numpy as np

SPLIT_TO_RELATIONS = {
    "train": [
        "P103",
        "P19",
        "P20",
        "P937",
        "P286",
        "P6",
    ],
    "validation": [
        "P159",
        "P364",
        "P108",
        "P488",
    ],
    "test": ["P36", "P449", "P39", "P264"],
}


if __name__ == "__main__":
    ds = load_dataset("coastalcph/fm_queries")
    ds["all_fm"] = ds["train"]
    rng = np.random.default_rng(7)
    print("Examples per relation", Counter(ds["all_fm"]["relation"]))
    for split in SPLIT_TO_RELATIONS.keys():
        ds[split] = ds["all_fm"].filter(
            lambda ex: ex["relation"] in SPLIT_TO_RELATIONS[split]
        )
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
    print(ds)
    print("Pushing to the hub...")
    ds.push_to_hub("coastalcph/fm_queries_classifier")
