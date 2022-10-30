import json
from pathlib import Path
from itertools import chain, combinations, product
from xml import dom
from torch import Tensor, sparse_coo_tensor, float64
from typing import Optional

# Covid19 Dataset Attributes
attributes = {"positive": 2, "gender": 2, "age": 4, "ethnicity": 8}
path = Path(__file__).resolve().parent.parent.joinpath("covid19_queries/queries.json")

# Query space size: 34425

def powerset(iter):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iter)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def create_queries():
    queries = []
    attr_values = []
    for _, domain_size in attributes.items():
        attr_values += [list(powerset(range(domain_size)))[1:]]  # Exclude empty set
    queries = list(product(*attr_values))

    query_tensors = []
    for query in queries:
        # print(query)
        query = [list(tup) for tup in query]
        query = [list(tup) for tup in product(*query)]
        query_tensors.append(query)
    # print(query_tensors)

    # Write queries to a json file
    queries = {}
    for i, query in enumerate(query_tensors):
        queries[i] = query
    
    json_object = json.dumps(queries, indent=4)

    # Writing to queries.json
    with open(path, "w") as outfile:
        outfile.write(json_object)


class Queries:
    def __init__(self, domain_size) -> None:
        self.domain_size = domain_size
        self.queries = None
        with open(path) as f:
            self.queries = json.load(f)
 
    def get_linear_query_tensor(self, query_id: int) -> Optional[Tensor]:
        if self.queries is None or query_id not in self.queries:
            return None
        query_tensor = self.queries[query_id]
        return sparse_coo_tensor(
            indices=query_tensor,
            values=[1.0] * len(query_tensor),   # add this as part of the query in queries.json
            size=(1, self.domain_size),
            dtype=float64,
        )

def main():
    create_queries()

if __name__ == "__main__":
    main()