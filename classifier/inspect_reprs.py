from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
import re
from functools import partial
from sklearn.manifold import TSNE
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/projects/nlp/data/constanzam/mutability_classifier/_projects_nlp_data_constanzam_llama_huggingface-ckpts_7B_07Jul_1630"
id2label = {1: "MUTABLE", 0: "IMMUTABLE"}
label2id = {"MUTABLE": 1, "IMMUTABLE": 0}
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

weight = model.score.state_dict()["weight"]
np.percentile(weight[1], np.arange(0, 101, 10))
# array([-6.83629885e-02, -2.67061600e-02, -1.73360482e-02, -1.06783467e-02,
#       -5.29242214e-03, -4.56801238e-06,  4.65411926e-03,  1.01346737e-02,
#        1.62944663e-02,  2.48887576e-02,  7.29644522e-02])

np.percentile(weight[0], np.arange(0, 101, 10))
# array([-0.07538875, -0.02487842, -0.01658648, -0.01079425, -0.00510834,
#        0.00029129,  0.00526351,  0.01119675,  0.01750649,  0.02644445,
#        0.06440652])

bigger = torch.zeros_like(weight, dtype=torch.bool)
bigger[0, :] = weight[0] > 0.026
bigger[1, :] = weight[1] > 0.024
torch.sum(bigger[0]), torch.sum(bigger[1]), torch.sum(bigger[0] & bigger[1])
# Out[150]: (tensor(423), tensor(442), tensor(50))
smaller = torch.zeros_like(weight, dtype=torch.bool)
smaller[0, :] = weight[0] < -0.024
smaller[1, :] = weight[1] < -0.026
torch.sum(smaller[0]), torch.sum(smaller[1]), torch.sum(smaller[0] & smaller[1])
# Out[25]: (tensor(436), tensor(440), tensor(40))

fig, ax = plt.subplots()
p = ax.pcolor(weight * bigger)
ax.set_yticks([0, 1])
fig.colorbar(p)
fig.savefig("output.png")
plt.close(fig)

fig, ax = plt.subplots()
p = ax.pcolor(weight * smaller, cmap="viridis_r")
ax.set_yticks([0, 1])
fig.colorbar(p)
fig.savefig("output.png")
plt.close(fig)

###############################################################################


def replace_subject(tokenizer, example):
    text = re.sub(r" \[Y\]\s?\.?$", "", example["template"].strip())
    text = text.replace("[X]", example["subject"]).strip()
    return {"text": text, **tokenizer(text, return_tensors="pt")}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/projects/nlp/data/constanzam/llama/huggingface-ckpts/7B"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
ds = load_dataset("cfierro/mutability_classifier_data", use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenized_ds = ds.map(partial(replace_subject, tokenizer))

rng = np.random.default_rng(42)
# train_sample = tokenized_ds["train"].select(
#    rng.choice(len(tokenized_ds["train"]), 50, replace=False)
# )
train_sample = tokenized_ds["train"]

last_token_reprs = []
for ex in tqdm(train_sample):
    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor(ex["input_ids"]).to(device),
            attention_mask=torch.tensor(ex["attention_mask"]).to(device),
            output_hidden_states=True,
        )
    last_token_reprs.append(outputs.hidden_states[-1][:, -1, :])

tsne = TSNE(
    n_components=2,
    perplexity=5,
    early_exaggeration=4.0,
    learning_rate=1000,
    n_iter=1000,
    n_iter_without_progress=50,
    min_grad_norm=0,
    method="exact",
    verbose=2,
)
Y = tsne.fit_transform(np.array([x[0].cpu().numpy() for x in last_token_reprs]))

cdict = {0: "red", 1: "blue"}
markers = [".", "1", "x", "v", "^", "<", "*", "s", "P"]
mdict = {
    relation: markers[i]
    for i, relation in enumerate(list(set(train_sample["relation"])))
}
fig1, ax1 = plt.subplots()
for relation in mdict.keys():
    rows = [ex["relation"] == relation for ex in train_sample]
    is_mutable = list(
        set(train_sample.filter(lambda ex: ex["relation"] == relation)["is_mutable"])
    )
    assert len(is_mutable) == 1
    ax1.scatter(
        Y[rows, 0],
        Y[rows, 1],
        color=cdict[is_mutable[0]],
        marker=mdict[relation],
    )
plt.close(fig1)
fig1.savefig("output.png")

################################# For all layers

import collections
import pandas as pd
import pickle

last_token_reprs = collections.defaultdict(list)
for ex in tqdm(train_sample):
    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor(ex["input_ids"]).to(device),
            attention_mask=torch.tensor(ex["attention_mask"]).to(device),
            output_hidden_states=True,
        )
    last_token_reprs["embed_output"].append(
        outputs.hidden_states[0][:, -1, :]
    ).cpu().numpy()
    for layer_i in range(1, len(outputs.hidden_states)):
        last_token_reprs[f"{layer_i}_output"].append(
            outputs.hidden_states[layer_i][:, -1, :].cpu().numpy()
        )

tsne = TSNE(
    n_components=2,
    perplexity=5,
    early_exaggeration=4.0,
    learning_rate=1000,
    n_iter=1000,
    n_iter_without_progress=50,
    min_grad_norm=0,
    method="exact",
    verbose=2,
)
cdict = {0: "red", 1: "blue"}
markers = [".", "1", "x", "v", "^", "<", "*", "s", "P"]
mdict = {
    relation: markers[i]
    for i, relation in enumerate(list(set(train_sample["relation"])))
}
df = pd.DataFrame(
    {"is_mutable": train_sample["is_mutable"], "relation": train_sample["relation"]}
)
for i, layer_name in enumerate(last_token_reprs.keys()):
    Y = tsne.fit_transform(np.array([x[0] for x in last_token_reprs[layer_name]]))
    pickle.dump(Y, open(f"tsne_{layer_name}.pickle", "wb"))

    fig1, ax1 = plt.subplots()
    for relation in mdict.keys():
        rows = [df.iloc[i]["relation"] == relation for i in range(len(df))]
        is_mutable = df[df["relation"] == relation]["is_mutable"].values[0]
        ax1.scatter(
            Y[rows, 0],
            Y[rows, 1],
            color=cdict[is_mutable],
            marker=mdict[relation],
            label=f"{relation}_{id2label[is_mutable]}",
        )
    ax1.set_title(f"output_{layer_name}")
    legend = ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.close(fig1)
    fig1.savefig(
        f"output_{layer_name}.png", bbox_extra_artists=(legend,), bbox_inches="tight"
    )
    #!imgcat output_{layer_name}.png


df_y = pd.DataFrame(Y)
df_y = pd.concat([df_y, df], axis=1)
df_y.groupby(by=[0, 1, "relation"]).size()


################################## PCA
from sklearn import decomposition

plt.cla()
pca = decomposition.PCA(n_components=2)
pca.fit(np.array([x[0][0].cpu().numpy() for x in last_token_reprs]))
X = pca.transform(np.array([x[0][0].cpu().numpy() for x in last_token_reprs]))

fig1, ax1 = plt.subplots()
for relation in mdict.keys():
    rows = [ex["relation"] == relation for ex in train_sample]
    is_mutable = list(
        set(train_sample.filter(lambda ex: ex["relation"] == relation)["is_mutable"])
    )
    assert len(is_mutable) == 1
    ax1.scatter(
        X[rows, 0],
        X[rows, 1],
        color=cdict[is_mutable[0]],
        marker=mdict[relation],
    )
plt.close(fig1)
fig1.savefig("output.png")

fig1 = plt.figure()
ax1 = plt.subplot(projection="3d")
for relation in mdict.keys():
    rows = [ex["relation"] == relation for ex in train_sample]
    is_mutable = list(
        set(train_sample.filter(lambda ex: ex["relation"] == relation)["is_mutable"])
    )
    assert len(is_mutable) == 1
    ax1.scatter(
        X[rows, 0],
        X[rows, 1],
        X[rows, 2],
        color=cdict[is_mutable[0]],
        marker=mdict[relation],
    )
fig1.savefig("output.png")
