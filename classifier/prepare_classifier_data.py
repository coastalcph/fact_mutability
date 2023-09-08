import pandas as pd
import re
from datasets import load_dataset, Dataset, Split, DatasetDict
import numpy as np

TEMPLAMA_R_TO_TEMPLATE = {
    # --- Train ---
    # Sports club -> person
    "P286": ("_X_ is the head coach of ", "[X]'s head coach is [Y]"),
    # Country -> person
    "P6": (
        "_X_ is the head of the government of ",
        "The head of government of [X] is [Y]",
    ),
    # Person -> political party
    "P102": (" is a member of the _X_", "[X] is a member of [Y]"),
    # --- Validation ---
    # Company -> person
    "P488": ("_X_ is the chair of ", "The role of chair at [X] is occupy by [Y]"),
    # Person -> company/university/organization
    "P108": (" works for _X_", "[X] works for [Y]"),
    # --- Test ---
    # Person -> polical position
    # This one is also present in LAMA but only with religious positions.
    "P39": (" holds the position of _X_", "[X] serves as [Y]"),
    # Person -> sports club
    "P54": (" plays for _X_", "[X] is associated with [Y]"),
}
SPLIT_TO_RELATIONS = {
    Split.TRAIN: ["P286", "P6", "P102", "P937", "P19", "P20", "P103", "P127"],
    Split.VALIDATION: ["P488", "P108", "P159", "P364"],
    Split.TEST: ["P39", "P54", "P131", "P36", "P449", "P413"],
}
LAMA_RELATIONS = [
    "P937",
    "P19",
    "P20",
    "P103",
    "P127",
    "P159",
    "P364",
    "P36",
    "P449",
    "P413",
]
MUTABLE_LAMA_RELATIONS = [
    "P937",
    # "P108", we could use it if we wanted to make the validation set bigger but
    # the objects in LAMA are not all that varied (~5-10 different).
]
# We sample from LAMA to have roughly the same number of examples than in
# TempLAMA.
SPLIT_TO_COUNT_PER_RELATION = {
    Split.TRAIN: 250,
}


def preprocess_templama_data():
    # (convert) P286 (Y is the head coach of X), P6 (Y is the head of government of X), P488 (Y is the chair of X)
    # > Train: 0.365 (1955)
    # 0.08 P6 (The head of government of X is Y)
    # 0.136 P102 (X is a member of Y)
    # 0.149 P286 (Y is the head coach of X) -> The head coach of Y is Y (person -> club)

    # > Val: 0.256 (1387)
    # 0.08 P488 (X's chair is Y) -> X occupies the role of chair at Y
    # 0.176 P108 (X works for Y -> X is employed by Y)

    # > Test: 0.374 (2000)
    # 0.187 P39 (X holds the position of Y -> "X serves as Y")
    # 0.187 P54 (X plays for Y) / X is associated with Y
    files = [
        "../data/templama/train.json",
        "../data/templama/val.json",
        "../data/templama/test.json",
    ]
    df = None
    for f in files:
        sub_df = pd.read_json(f, lines=True)
        df = pd.concat([df, sub_df]) if df is not None else sub_df
    df = df[~df["relation"].isin(["P69", "P127"])]
    df["sub_uri"] = df.apply(lambda x: x["id"].split("_")[0], axis=1)
    df_agg = (
        df[["query", "answer", "id", "relation", "sub_uri"]]
        .groupby(["relation", "query"], as_index=False)
        .agg({"answer": list, "id": list, "sub_uri": set})
    )
    answers_sets = []
    for ids, answers_per_year in zip(df_agg.id, df_agg.answer):
        example_answers = set()
        for _, answers in zip(ids, answers_per_year):
            for a in answers:
                example_answers.add(a["name"])
        answers_sets.append(list(example_answers))
    df_agg["answers"] = answers_sets
    templates = []
    subjects = []
    for query, relation in zip(df_agg["query"], df_agg.relation):
        query = re.sub(r"\.?$", "", query)
        template, final_template = TEMPLAMA_R_TO_TEMPLATE[relation]
        template_start = query.find(template)
        if template_start == 0:
            subject = query[len(template) :]
        else:
            subject = query[0:template_start]
        templates.append(final_template)
        subjects.append(subject)
    df_agg["template"] = templates
    df_agg["subject"] = subjects
    df_agg["is_mutable"] = 1
    df_agg["sub_uri"] = [list(ids) for ids in df_agg.sub_uri]
    return df_agg[
        ["relation", "template", "subject", "answers", "is_mutable", "sub_uri"]
    ]


def preprocess_pararel_data():
    # Pararel relations
    # 250 of each relation in train
    # P19 (X was born in Y) (people - country) {should we add a mutable relation with country? like lives in? P937 X works in the city of Y}
    # P20 (X died in Y)
    # P103 (The native language of X is Y)
    # P127 (X is developed by Y) (software - company)

    # For test we should be ok to use types that are not in mutable to see if we generalize to those
    # Val
    # P159 (The headquarter of X is in Y)
    # P364 (The original language of [X] is [Y].)
    # Test
    # P131 (X is located in) (district/street/natural places - city)
    # P36 (The capital of [X] is [Y])
    # P449 (X was originally aired on Y)
    # P413 (X plays in Y position)
    ds = load_dataset("cfierro/pararel_data", use_auth_token=True)
    pararel = pd.DataFrame(
        {
            k: ds["train"][k]
            for k in ["relation", "template", "subject", "object", "sub_uri"]
        }
    )
    pararel["template_obj@end_removed"] = pararel.apply(
        lambda x: re.sub(r" \[Y\]\s?\.?$", "", x["template"].strip()), axis=1
    )
    pararel["autoregressive"] = pararel.apply(
        lambda x: "[Y]" not in x["template_obj@end_removed"], axis=1
    )
    pararel = pararel[
        (pararel.autoregressive) & (pararel.relation.isin(LAMA_RELATIONS))
    ]
    # If we have constraint, then we select K subjects at random.
    for split, max_items in SPLIT_TO_COUNT_PER_RELATION.items():
        for relation in SPLIT_TO_RELATIONS[split]:
            if relation not in LAMA_RELATIONS:
                continue
            subjects = pararel[pararel.relation == relation].subject.unique()
            if max_items < len(subjects):
                to_remove = np.random.choice(len(subjects), len(subjects) - max_items)
                subjects_to_remove = subjects[to_remove]
            else:
                subjects_to_remove = []
                print(
                    "Warning. For relation {} we're going to select {} items, "
                    "which is less than expected ({})".format(
                        relation, len(subjects), max_items
                    )
                )
            pararel.drop(
                pararel[
                    (pararel.relation == relation)
                    & (pararel.subject.isin(subjects_to_remove))
                ].index,
                inplace=True,
            )
    # We only select one template per subject.
    pararel["selected"] = False
    for relation in LAMA_RELATIONS:
        templates = pararel[pararel.relation == relation].template.unique()
        if len(templates) == 1:
            pararel[pararel.relation == relation]["selected"] = True
            continue
        for subject in pararel[pararel.relation == relation].subject.unique():
            selected = np.random.choice(len(templates), 1)
            template_selected = templates[selected][0]
            pararel.loc[
                (pararel.relation == relation)
                & (pararel.subject == subject)
                & (pararel.template == template_selected),
                "selected",
            ] = True
    pararel = pararel[pararel.selected]
    pararel["is_mutable"] = 0
    pararel.loc[pararel.relation.isin(MUTABLE_LAMA_RELATIONS), "is_mutable"] = 1
    pararel["answers"] = [[object_] for object_ in pararel.object]
    pararel["sub_uri"] = [[uri] for uri in pararel.sub_uri]
    return pararel[
        ["relation", "template", "subject", "answers", "is_mutable", "sub_uri"]
    ]


if __name__ == "__main__":
    templama_data = preprocess_templama_data()
    lama_data = preprocess_pararel_data()
    df = pd.concat([templama_data, lama_data])
    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(
                df[df.relation.isin(SPLIT_TO_RELATIONS[Split.TRAIN])],
                preserve_index=False,
            ),
            "validation": Dataset.from_pandas(
                df[df.relation.isin(SPLIT_TO_RELATIONS[Split.VALIDATION])],
                preserve_index=False,
            ),
            "test": Dataset.from_pandas(
                df[df.relation.isin(SPLIT_TO_RELATIONS[Split.TEST])],
                preserve_index=False,
            ),
        }
    )
    print("Pushing to the hub...")
    ds.push_to_hub("cfierro/mutability_classifier_data", private=True)
