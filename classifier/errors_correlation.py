import pandas as pd
import argparse
from datasets import load_dataset
from scipy.stats import pearsonr
import torch
import wandb


def check_ds_examples_match(lm_preds, clf_preds):
    lm_ids = set(lm_preds["id"].values)
    clf_ids = set(clf_preds["id"].values)
    ds = load_dataset("cfierro/mutability_classifier_data", use_auth_token=True)
    for ex in ds["validation"]:
        if len(ex["sub_uri"]) > 1:
            ids = ["{}_{}".format(sub_uri, ex["relation"]) for sub_uri in ex["sub_uri"]]
            assert len(lm_ids.intersection(set(ids))) == 1
            assert list(lm_ids.intersection(set(ids)))[0] in clf_ids
        else:
            id = "{}_{}".format(ex["sub_uri"][0], ex["relation"])
            assert id in lm_ids, id
            assert id in clf_ids, id


def main(args):
    clf_preds = pd.read_json(args.clf_predictions_path)
    lm_preds = pd.read_json(args.lm_predictions_path)

    clf_data = []
    clf_preds["prob"] = clf_preds.apply(
        lambda x: torch.softmax(torch.tensor(x["pred_score"]), axis=0)[
            x["prediction"]
        ].item(),
        axis=1,
    )
    for i in range(len(clf_preds)):
        row = clf_preds.iloc[i]
        data = row["input"]
        # lm_preds should contain one of these sub_uri.
        for sub_uri in data["sub_uri"]:
            query_id = "{}_{}".format(sub_uri, data["relation"])
            clf_data.append(
                (
                    query_id,
                    row["prediction"],
                    row["prob"],
                    row["label"],
                    row["relation"],
                )
            )
    clf_df = pd.DataFrame(
        clf_data, columns=["id", "clf_label_pred", "clf_prob", "is_mutable", "relation"]
    )
    clf_df["clf_correct"] = (clf_df["clf_pred"] == clf_df["is_mutable"]).astype(int)
    check_ds_examples_match(lm_preds, clf_df)
    df = pd.merge(lm_preds, clf_df, on="id")
    corrs = {}
    for k1, k2 in [("f1", "clf_label_pred"), ("f1", "clf_correct")]:
        corr = pearsonr(df[k1].values, df[k2].values)
        print(f"Pearson {k1}-{k2}", corr)
        corrs.update(
            {f"corr/{k1}-{k2}": corr.statistic, f"pvalue/{k1}-{k2}": corr.pvalue}
        )
    wandb.log(corrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--lm_predictions_path",
        required=True,
        type=str,
        help="Path to predictions",
    )
    parser.add_argument(
        "--clf_predictions_path",
        type=str,
        required=True,
        help="Path to predictions",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Path to predictions",
    )
    args = parser.parse_args()

    project_name = "lm_probing_clf_corr"
    wandb.init(
        project=project_name,
        name=args.exp_name,
        config=args,
    )

    main(args)
