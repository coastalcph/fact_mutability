import numpy as np
import json
from dataset import relations, relations_by_type, sorted_relations_by_type

def main():
    for type, rels in sorted_relations_by_type.items():
        for rel in rels:
            num_objects = list()
            subjects = json.load(open("./data/wikidata/objects_by_freq/{}.json".format(rel)))
            for s in subjects:
                num_objects.append(len(s['objects']))
            print(np.mean(num_objects))
        print()




if __name__ == '__main__':
    main()