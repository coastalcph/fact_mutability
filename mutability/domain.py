from collections import defaultdict
from typing import List, Optional, Dict

import matplotlib.pyplot as plt

class Answer:
    def __init__(self, texts: List[str], year: int):
        self.texts = texts
        self.year = year

    def __repr__(self):
        texts = '\n'.join(self.texts)
        return f"({self.year})\t{texts}"

    def dump(self) -> dict:
        return {
            "text": self.texts,
            "year": self.year
        }


class Query:
    def __init__(self, id: str, text: str, answers: List[Answer]):
        self.id = id
        self.text = text
        self.answers = answers
        self.ratio = self.__compute_ratio()

    def __repr__(self):
        t = f"{self.text}\n"
        t += "\n".join([str(a) for a in self.answers])
        return t

    def __compute_ratio(self):
        # TODO: Better capture mutability over years by keeping track of answers
        unique_names = set()
        years = set()
        answers = self.group_answers_by_year()
        max_answers_a_year = max([len(a) for _, a in answers.items()])
        min_answers_a_year = min([len(a) for _, a in answers.items()])
        for year, answers_a_year in answers.items():
            for answer in answers_a_year:
                unique_names.add(answer.texts[0])
            years.add(year)
        ratio = (len(unique_names) - (max_answers_a_year - min_answers_a_year)) / len(years)
        return ratio
    
    def group_answers_by_year(self):
        answers_by_year = defaultdict(list)
        for answer in self.answers:
            answers_by_year[answer.year].append(answer)
        return answers_by_year

    def get_ratio(self) -> float:
        return self.ratio

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
        if query in self.queries:
            return self.queries[id]
        else:
            raise IndexError(f'Query "{id}" does not exists in dataset')

    def plot_ratios(self):
        ratios = [q.get_ratio() for q in self]
        plt.hist(ratios)
        plt.show()
