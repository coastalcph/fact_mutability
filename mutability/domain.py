from typing import List, Optional, Dict

import matplotlib.pyplot as plt

class Answer:
    def __init__(self, text: str, year: int):
        self.text = text
        self.year = year

    def __repr__(self):
        return f"({self.year})\t{self.text}"

    def dump(self) -> dict:
        return {
            "text": self.text,
            "year": self.year
        }


class Query:
    def __init__(self, text: str, answers: List[Answer]):
        self.text = text
        self.answers = answers
        self.ratio = self.__compute_ratio()

    def __repr__(self):
        t = f"{self.text}\n"
        t += "\n".join([str(a) for a in self.answers])
        return t

    def __compute_ratio(self):
        unique_names = set()
        years = set()
        for answer in self.answers:
            unique_names.add(answer.text)
            years.add(answer.year)
        ratio = (len(unique_names) - 1) / len(years)
        return ratio

    def get_ratio(self) -> float:
        return self.ratio

    def dump(self) -> dict:
        return {
            "text": self.text,
            "ratio": self.ratio,
            "answers": [a.dump() for a in self.answers]
        }


class Queries:
    def __init__(self, queries: Optional[Dict[str, Query]]=None):
        self.queries = queries if queries else dict()
    
    def add_query(self, query: Query):
        self.queries[query.text] = query

    def __iter__(self):
        for each in self.queries.values():
            yield each
    
    def __getitem__(self, query: str):
        if query in self.queries:
            return self.queries[query]
        else:
            raise IndexError(f'Query "{query}" does not exists in dataset')

    def plot_ratios(self):
        ratios = [q.get_ratio() for q in self]
        plt.hist(ratios)
        plt.show()

