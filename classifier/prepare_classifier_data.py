from datasets import load_dataset
from collections import Counter
import numpy as np
import itertools
import collections

SPLIT_TO_RELATIONS = {
    "train": [
        "P103",
        "P19",
        "P937",
        "P286",
        "P6",
        "P159",
        "P27",
        "P1412",
        "P190",
        "P140",
    ],
    "validation": ["P20", "P364", "P108", "P488", "P69", "P101"],
    "test": ["P36", "P449", "P39", "P264", "P47", "P136"],
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
    ds.pop("all_fm")
    ds = ds.filter(lambda ex: len(ex["answer"]) > 0)
    ds = ds.map(lambda ex: {"is_mutable": 1 if ex["type"] == "mutable" else 0})
    ds_1_n = ds.filter(lambda ex: ex["type"] != "immutable")
    ds_1_1 = ds.filter(lambda ex: ex["type"] != "immutable_n")

    for ds_name, ds in [("immutable-1-n", ds_1_n), ("immutable-1-1", ds_1_1)]:
        print(ds_name, ds)
        split_to_subjs = collections.defaultdict(dict)
        split_to_objs = collections.defaultdict(dict)
        mut_types = list(set(ds[split]["type"]))
        for split in ds.keys():
            print("Split statistics")
            counts = Counter(ds[split]["type"])
            print(counts, {k: c / len(ds[split]) for k, c in counts.items()})
            counts = Counter(ds[split]["relation"])
            print(counts, {k: c / len(ds[split]) for k, c in counts.items()})
            for mut_type in mut_types:
                split_to_subjs[split][mut_type] = set(
                    [
                        id_.split("_")[0]
                        for id_, t in zip(ds[split]["id"], ds[split]["type"])
                        if t == mut_type
                    ]
                )
                split_to_objs[split][mut_type] = set(
                    [
                        answers[0]["wikidata_id"]
                        for answers, t in zip(ds[split]["answer"], ds[split]["type"])
                        if t == mut_type
                        # for a in answers
                    ]
                )

        remove_subjs = set()
        remove_objs = set()
        for split_i, split_j in itertools.combinations(SPLIT_TO_RELATIONS.keys(), 2):
            print(split_i, len(ds[split_i]), "-", split_j, len(ds[split_j]))
            for set_name, remove_set, split_to_items in [
                ("subjs", remove_subjs, split_to_subjs),
                ("objs", remove_objs, split_to_objs),
            ]:
                print(set_name)
                items_inters0 = split_to_items[split_i][mut_types[0]].intersection(
                    split_to_items[split_j][mut_types[0]]
                )
                print("intersection", mut_types[0], len(items_inters0))
                items_inters1 = split_to_items[split_i][mut_types[1]].intersection(
                    split_to_items[split_j][mut_types[1]]
                )
                print("intersection", mut_types[1], len(items_inters1))
                # We only care about the subjects that just one of them has in
                # different splits. If both have it the two splits then there
                # should be no problem of it being a signal.
                items_inters0 = items_inters0.difference(
                    split_to_items[split_i][mut_types[1]].union(
                        split_to_items[split_j][mut_types[1]]
                    )
                )
                items_inters1 = items_inters1.difference(
                    split_to_items[split_i][mut_types[0]].union(
                        split_to_items[split_j][mut_types[0]]
                    )
                )
                print("Need to be removed", len(items_inters0.union(items_inters1)))
                remove_set.update(items_inters0.union(items_inters1))
                print("len remove_set", len(remove_set))

        for split in ["train", "test"]:
            ds[split] = ds[split].filter(
                lambda ex: ex["id"].split("_")[0] not in remove_subjs
                and not ex["answer"][0]["wikidata_id"] in remove_objs
                # and not set(
                #    [answer["wikidata_id"] for answer in ex["answer"]]
                # ).intersection(remove_objs)
            )
        print(f"-------- Final summary {ds_name} --------")
        print(ds)
        print("Split statistics")
        for split in ds.keys():
            print(split)
            counts = Counter(ds[split]["type"])
            print(counts, {k: c / len(ds[split]) for k, c in counts.items()})
            counts = Counter(ds[split]["relation"])
            print(counts, {k: c / len(ds[split]) for k, c in counts.items()})
        print("-----------------------------")

    print("Dataset mutable (1-1) vs immutable", ds_1_1)
    print("Dataset mutable (1-N) vs immutable", ds_1_n)
    print("Pushing to the hub...")
    ds_1_1.push_to_hub("coastalcph/fm_classifier-1-1")
    ds_1_n.push_to_hub("coastalcph/fm_classifier-1-n")
