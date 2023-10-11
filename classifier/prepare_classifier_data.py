from datasets import load_dataset
from collections import Counter

split_to_relation = {
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
    ds["all"] = ds["train"]
    print("Examples per relation", Counter(ds["all"]["relation"]))
    for split in split_to_relation.keys():
        ds[split] = ds["all"].filter(
            lambda ex: ex["relation"] in split_to_relation[split]
        )
        assert set(ds[split]["type"]).intersection(["immutable_n"]) == set()
    ds = ds.map(lambda ex: {"is_mutable": 1 if ex["type"] == "mutable" else 0})
    print(ds)
    print("Pushing to the hub...")
    ds.push_to_hub("coastalcph/fm_queries_classifier")
