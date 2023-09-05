import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from utils.data_handling import *

def main():
    model = "flan-t5"
    
    # Mutable
    mutable_dataset = build_dataset('data/val_with_aliases.json')
    predictions_mu = load_predictions(f"data/predictions-val-{model}.json")
    c_mut = json.load(open("./data/cooccurrences_mut.json"))

    # Homemade
    immutable_dataset = build_dataset('data/immutable_with_aliases.json')
    predictions_imm = load_predictions(f"data/predictions-imm-{model}.json")
    c_imm = json.load(open("./data/cooccurrences_imm.json"))

    # LAMA
    lama_dataset = build_dataset('data/lama_with_aliases.json')
    predictions_lama = load_predictions(f"data/predictions-lama-{model}.json")
    c_lama = json.load(open("./data/cooccurrences_lama.json"))

    occurrences = list()
    predictions = list()

    imm_occ = list()
    imm_preds = list()

    lama_occ = list()
    lama_preds = list()

    mut_occ = list()
    mut_preds = list()

    for query in tqdm(immutable_dataset):
        prediction = get_prediction(predictions_imm, query.id)
        subj, relation = query.id.split("_") 
        answers = query.answers
        for answer in answers:
            key = f"{subj}-{answer.qcode}"
            if key in c_imm and prediction:
                confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
                confidences = [c for c in confidences if not np.isnan(c)]
                confidence = confidences[0]
                imm_preds.append(confidence)
                c = c_imm[key]
                imm_occ.append(c)
                occurrences.append(c)
                predictions.append(confidence)

    for query in tqdm(lama_dataset):
        prediction = get_prediction(predictions_lama, query.id)
        subj, relation = query.id.split("_") 
        answers = query.answers
        for answer in answers:
            key = f"{subj}-{answer.qcode}"
            if key in c_lama and prediction:
                confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
                confidences = [c for c in confidences if not np.isnan(c)]
                confidence = confidences[0]
                lama_preds.append(confidence)
                c = c_lama[key]
                occurrences.append(c)
                lama_occ.append(c)
                predictions.append(confidence)

    for query in tqdm(mutable_dataset):
        prediction = get_prediction(predictions_mu, query.id)
        subj, relation = query.id.split("_") 
        answers = query.answers
        for answer in answers:
            key = f"{subj}-{answer.qcode}"
            if key in c_mut and prediction:
                confidences = sorted([p['first_token_probability'] for p in prediction['predictions']], reverse=True)
                confidences = [c for c in confidences if not np.isnan(c)]
                confidence = confidences[0]
                mut_preds.append(confidence)
                c = c_mut[key]
                occurrences.append(c)
                mut_occ.append(c)
                predictions.append(confidence)

    print(len(imm_occ), len(imm_preds))
    print(len(lama_occ), len(lama_preds))
    print(len(mut_occ), len(mut_preds))

    print(pearsonr(np.log(occurrences), predictions))

    plt.scatter(np.log(imm_occ), imm_preds, alpha=0.5, label='imm')
    plt.scatter(np.log(lama_occ), lama_preds, alpha=0.5, label='lama')
    plt.scatter(np.log(mut_occ), mut_preds, alpha=0.5, label='mut')

    # plt.hist(np.log(imm_occ), alpha=0.5, label='imm')
    # plt.hist(np.log(lama_occ), alpha=0.5, label='lama')
    # plt.hist(np.log(mut_occ), alpha=0.5, label='mut')

    plt.legend(loc='upper right')

    plt.show()




if __name__ == '__main__':
    main()