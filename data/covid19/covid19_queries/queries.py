import json
import math
from itertools import chain, combinations, product
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from precycle.budget.histogram import (
    build_sparse_tensor,
    k_way_marginal_query_list,
)
from precycle.utils.utils import REPO_ROOT

app = typer.Typer()

# Note: this file is independent to the covid dataset - applies to all datasets


def powerset(iter):
    s = list(iter)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# TODO: Prune the query space - not all queries are important
def create_all_queries(queries_path, attributes_domain_sizes):
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
    with open(queries_path, "w") as outfile:
        outfile.write(json_object)


def create_specific_queries(
    queries_path, attributes_domain_sizes, type="positive_cases"
):
    queries = []
    attributes_domain_sizes = [2, 2, 4, 8]

    if type == "positive_cases":
        queries = [[[1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]]]
        print(queries)
        query_tensors = []
        for query in queries:
            query = [list(tup) for tup in query]
            query = [list(tup) for tup in product(*query)]
            query_tensors.append(query)
    elif type == "2way_marginal_mix":
        query_tensors = []
        dicts = [
            {0: 1, 1: 0},
            {0: 1, 1: 1},
            {0: 1, 2: 0},
            {0: 1, 2: 1},
            {1: 0, 2: 0},
            {1: 1, 2: 1},
        ]
        for d in dicts:
            query_tensors.append(k_way_marginal_query_list(d, attributes_domain_sizes))
    elif type == "all_2way_marginals":
        query_tensors = []
        dicts = []

        # Conjunctions of 0 with something else, then 1 with something else than 0, etc.
        for i in range(4):
            for a in range(attributes_domain_sizes[i]):
                for j in range(i + 1, 4):
                    for b in range(attributes_domain_sizes[j]):
                        print(f"{i}: {a}, {j}: {b}")
                        dicts.append({i: a, j: b})

        # Only 84 2-way marginals! Pretty small, let's see if that's enough for PMW to shine
        print(f"We have {len(dicts)} 2-way marginals. Computing the tensors...")

        for d in dicts:
            query_tensors.append(k_way_marginal_query_list(d, attributes_domain_sizes))

    elif type == "all_3way_marginals":
        query_tensors = []
        dicts = []

        # Conjunctions of 0 with something else, then 1 with something else than 0, etc.
        for i in range(4):
            for a in range(attributes_domain_sizes[i]):
                for j in range(i + 1, 4):
                    for b in range(attributes_domain_sizes[j]):
                        for k in range(j + 1, 4):
                            for c in range(attributes_domain_sizes[k]):
                                print(f"{i}: {a}, {j}: {b}, {k}: {c}")
                                dicts.append({i: a, j: b, k: c})

        # 176 only
        print(f"We have {len(dicts)} 3-way marginals. Computing the tensors...")

        for d in dicts:
            query_tensors.append(k_way_marginal_query_list(d, attributes_domain_sizes))

    else:
        num_queries = int(type)
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

        assert num_queries <= len(query_tensors)
        print(
            f"Keeping {num_queries} queries from the total {len(query_tensors)} queries."
        )

        query_tensors = query_tensors[0:num_queries]

    # Write queries to a json file
    queries = {}
    for i, query in enumerate(query_tensors):
        queries[i] = query

    json_object = json.dumps(queries, indent=4)

    # Writing to queries.json
    with open(queries_path, "w") as outfile:
        outfile.write(json_object)


class QueryPool:
    def __init__(self, attribute_domain_sizes, queries_path) -> None:
        self.attribute_domain_sizes = attribute_domain_sizes
        self.domain_size = math.prod(attribute_domain_sizes)
        self.queries = None
        with open(queries_path) as f:
            self.queries = json.load(f)

    def get_query_sql(self, f, query_id: int) -> Optional[str]:
        # TODO: allow sql like queries too
        pass

    def get_query(self, query_id: int):
        query_id_str = str(query_id)
        if query_id_str in self.queries:
            query_tensor = self.queries[query_id_str]
            # print(query_tensor)
            query = build_sparse_tensor(
                bin_indices=query_tensor,
                values=[1.0]
                * len(
                    query_tensor
                ),  # TODO: add values as part of the query in queries.json
                attribute_sizes=self.attribute_domain_sizes,
            )
        else:
            logger.error("Not implemented")
            exit(1)  # not implemented for now
            query = self.get_query_sql(query_id_str)

        assert query is not None
        return query


def main(
    queries_dir: str = REPO_ROOT.joinpath("data/covid19/covid19_queries"),
    blocks_metadata_path: str = REPO_ROOT.joinpath(
        "data/covid19/covid19_data/metadata.json"
    ),
    workload="all_2way_marginals",
):

    try:
        with open(blocks_metadata_path) as f:
            blocks_metadata = json.load(f)
    except NameError:
        logger.error("Dataset metadata must have be created first..")
        exit(1)

    attributes_domain_sizes = blocks_metadata["attributes_domain_sizes"]
    # create_all_queries(
    #     queries_path,
    #     blocks_metadata["attributes_domain_sizes"]
    # )  # Query space size for covid dataset: 34425

    # create_specific_queries(queries_path, attributes_domain_sizes)
    # create_specific_queries(queries_path, attributes_domain_sizes, type="2way_marginal_mix")
    queries_path = Path(queries_dir).joinpath(f"{workload}.queries.json")
    create_specific_queries(queries_path, attributes_domain_sizes, type=workload)


if __name__ == "__main__":
    typer.run(main)
