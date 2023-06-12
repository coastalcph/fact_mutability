from collections import defaultdict
from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

class Relation:
    def __init__(self, id, mutation_mean, mutation_std) -> None:
        self.id = id
        self.mutation_mean = mutation_mean
        self.mutation_std = mutation_std
    
    def sample_mutation_rate(self) -> float:
        return max(0, np.random.normal(self.mutation_mean, self.mutation_std, 1)[0])


class Answer:
    def __init__(self, texts: List[str], year: int, qcode: str):
        self.texts = sorted(texts)
        self.year = year
        self.qcode = qcode

    def __repr__(self):
        texts = '\n'.join(self.texts)
        return f"({self.year})\t{texts}"

    def dump(self) -> dict:
        return {
            "text": self.texts,
            "year": self.year,
            "qcode": self.qcode
        }


class Query:
    def __init__(self, id: str, text: str, answers: List[Answer], relation_id: str):
        self.id = id
        self.text = text
        self.answers = answers
        self.ratio = self.__compute_ratio()
        self.relation_id = relation_id

    def __repr__(self):
        t = f"{self.text}\n"
        t += "\n".join([str(a) for a in self.answers])
        return t

    def __compute_ratio(self):
        years = set()
        answers = self.group_answers_by_year()
        current_answer = set()
        changes = 0
        for year, answers_a_year in answers.items():
            years.add(year)
            new_answer = {a.texts[0] for a in answers_a_year}
            if len(current_answer):  # check if the answer changed
                if new_answer != current_answer:  # something changed
                    changes += 1 
            current_answer = new_answer
        ratio = (changes) / len(years)
        return ratio

    def group_answers_by_year(self):
        answers_by_year = defaultdict(list)
        for answer in self.answers:
            answers_by_year[answer.year].append(answer)
        return answers_by_year

    def get_ratio(self) -> float:
        return self.ratio

    def get_most_recent_answer(self):
        answers_by_year = self.group_answers_by_year()
        return sum([a.texts for a in answers_by_year[sorted(list(answers_by_year.keys()))[-1]]], [])

    def get_most_frequent_answer(self):
        qcodes = [a.qcode for a in self.answers]
        most_frequent_qcode = max(set(qcodes), key=qcodes.count)
        for a in sorted(self.answers, key=lambda x: x.year)[::-1]:
            if a.qcode == most_frequent_qcode:
                return a.texts

    def get_answer_by_year(self, year):
        answers_for_year = self.group_answers_by_year().get(year, None)
        if answers_for_year is None:
            return None
        else:
            return sum([a.texts for a in answers_for_year], [])

    def get_relevant_target(self, target_mode):
        if target_mode == 'most_recent':
            return self.get_most_recent_answer()
        elif target_mode == 'most_frequent':
            return self.get_most_frequent_answer()
        elif target_mode.isnumeric():
            return self.get_answer_by_year(target_mode)

    def dump(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "ratio": self.ratio,
            "answers": [a.dump() for a in self.answers]
        }


class Queries:
    def __init__(self, queries: Optional[Dict[str, Query]]=None):
        self.queries = queries if queries else dict()
    
    def add_query(self, query: Query):
        self.queries[query.id] = query

    def __iter__(self):
        for each in self.queries.values():
            yield each
    
    def __getitem__(self, id: str):
        if id in self.queries:
            return self.queries[id]
        else:
            raise IndexError(f'Query "{id}" does not exists in dataset')

    def plot_ratios(self):
        ratios = [q.get_ratio() for q in self]
        plt.hist(ratios)
        plt.show()

