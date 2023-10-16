import json
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = './data/wikidata/objects_by_freq'
    objects = os.listdir(path)
    files = [f for f in objects if 'json' in f]
    mutations = list()
    for f in files:
        mutation = dict()
        mutation['rel_id'] = f.split('.')[0]
        data = json.load(open(os.path.join(path, f)))
        mutation['num_objects'] = [len(d['objects']) for d in data]
        mutation['relation_name'] = data[0]['relation']
        mutation['avg'] = np.mean(mutation['num_objects'])
        mutations.append(mutation)
    sorted_mutations = sorted(mutations, key=lambda x: x['avg'])
    for mutation in sorted_mutations:
        mu = np.mean(mutation['num_objects'])
        sigma = np.std(mutation['num_objects'])
        rel = mutation['relation_name']
        rel_id = mutation['rel_id']
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(mutation['num_objects'], 20, density=True)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
        ax.plot(bins, y, '--')
        ax.set_xlabel('Num. Objects')
        ax.set_ylabel('Probability density')
        ax.set_title(f"{rel} ({rel_id}): " fr'$\mu={mu:.0f}$, $\sigma={sigma:.0f}$')
        fig.tight_layout()
        # plt.show()
        print(rel)
        print(rel_id)
        print(mu)
        print(sigma)
        print()


if __name__ == '__main__':
    main()