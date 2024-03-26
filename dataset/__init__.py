from collections import defaultdict

def sort_relations(rels):
    return sorted(rels, key=lambda x: int(x.split("P")[1]))

relations = {
    "P740": "immutable",
    "P103": "immutable",
    "P19": "immutable",
    "P20": "immutable",
    "P30": "immutable",
    "P36": "immutable",
    "P159": "immutable",
    "P449": "immutable",
    "P364": "immutable",
    "P495": "immutable",
    "P140": "immutable",
    "P138": "immutable",
    "P47": "immutable_n",
    "P136": "immutable_n",
    "P166": "immutable_n",
    "P69": "immutable_n",
    "P101": "immutable_n",
    "P530": "immutable_n",
    "P27": "immutable_n",
    "P1412": "immutable_n",
    "P1303": "immutable_n",
    "P190": "immutable_n",
    "P937": "mutable",
    "P108": "mutable",
    "P488": "mutable",
    "P286": "mutable",
    "P6": "mutable",
    "P39": "mutable",
    "P54": "mutable",
    "P264": "mutable",
    "P551": "mutable",
    "P451": "mutable",
    "P1308": "mutable",
    "P210": "mutable",
    "P1037": "mutable",
}

relations_by_type = defaultdict(list)
for rel, type in relations.items():
    relations_by_type[type].append(rel)

sorted_relations_by_type = dict()
for type, rels in relations_by_type.items():
    sorted_relations_by_type[type] = sort_relations(rels)