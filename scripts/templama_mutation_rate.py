from utils.data_handling import *
from analysis.analyze_inference import build_relations

def main():
    train = build_dataset('data/templama/train_with_aliases.json')
    val = build_dataset('data/templama/val_with_aliases.json')
    for query in val:
        train.add_query(query)
    test = build_dataset('data/templama/test_with_aliases.json')
    for query in test:
        train.add_query(query)
    print(len(train))
    train.plot_ratios()

    relations = build_relations(train)
    for relation_id, relation in relations.items():
        print(relation_id, relation.mutation_mean, relation.mutation_std)

if __name__ == '__main__':
    main()