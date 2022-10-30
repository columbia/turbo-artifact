import json
from pathlib import Path
from itertools import chain, combinations, product
from xml import dom
from torch import Tensor, sparse_coo_tensor, float64
from typing import Optional
from loguru import logger
from torch import Tensor
from pandas import DataFrame

path = Path(__file__).resolve().parent.parent.joinpath("covid19_queries/queries.json")
blocks_metadata_path = Path(__file__).resolve().parent.parent.joinpath("covid19_data/metadata.json")

def powerset(iter):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iter)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def create_queries(attributes_domain_sizes):
    queries = []
    attr_values = []
    for domain_size in attributes_domain_sizes:
        attr_values += [list(powerset(range(domain_size)))[1:]]  # Exclude empty set
    queries = list(product(*attr_values))

    query_tensors = []
    for query in queries:
        query = [list(tup) for tup in query]
        query = [list(tup) for tup in product(*query)]
        query_tensors.append(query)

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
 
    def get_query_sql(self, f, query_id: int) -> Optional[str]:
        # TODO: allow sql like queries too
        pass

    def get_query_tensor(self, query_id: int) -> Optional[Tensor]:
        if self.queries is None or query_id not in self.queries:
            return None
        query_tensor = self.queries[query_id]
        return sparse_coo_tensor(
            indices=query_tensor,
            values=[1.0] * len(query_tensor),   # add this as part of the query in queries.json
            size=(1, self.domain_size),
            dtype=float64,
        )

    def get_query(self, query_id: int):
        if True:
            query = Tensor(self.get_query_tensor(query_id))
        else:
            query = DataFrame(get_query_sql(query_id))
        
        assert query is not None
        return query
        


def main():
    try:
        f = open(blocks_metadata_path)
        blocks_metadata = json.load(f)
    except: 
        logger.error("Dataset metadata must have been created first..")
        exit(1)
    finally:
        f.close()

    create_queries(blocks_metadata['attributes_domain_sizes'])  # Query space size: 34425

if __name__ == "__main__":
    main()