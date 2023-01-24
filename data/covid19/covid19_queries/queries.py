import json
from itertools import chain, combinations, product
from pathlib import Path

import typer
from loguru import logger

from precycle.budget.histogram import k_way_marginal_query_list
from precycle.utils.utils import REPO_ROOT

app = typer.Typer()


def powerset(iter):
    s = list(iter)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


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

    elif type == "8":
        # number of people who tested positive AND are females AND 18-49 AND any ethnicity
        q1 = [[1], [1], [1], [0, 1, 2, 3, 4, 5, 6, 7]]
        # are positive AND are either female or male AND are either one of the 4 age groups AND are either one of the 8 ethnicities
        q2 = [[1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]]
        # got tested AND male or female and 18-49 and white
        q3 = [[0, 1], [0, 1], [1], [6]]
        # number of people who tested positive AND are male AND 50-64 AND American Indian, Latino, Black,  or Multi-Race
        q4 = [[1], [0], [2], [0, 2, 3, 7]]
        # number of people who tested positive AND are female AND any age AND American Indian, Latino, Black, or Multi-Race
        q5 = [[1], [1], [0, 1, 2, 3], [0, 2, 3, 7]]
        # number of people who tested negative AND are male AND 18-49 AND white or asian
        q6 = [[0], [0], [1], [1, 6]]
        # got tested AND male and 18-49 and white
        q7 = [[0, 1], [0], [1], [1, 6]]
        # number of people who got tested  AND are female AND any age AND American Indian, Latino, Black, or Multi-Race
        q8 = [[[0, 1], [1], [0, 1, 2, 3], [0, 2, 3, 7]]]
        queries = [q1, q2, q3, q4, q5, q6, q7, q8]
        query_tensors = []
        for query in queries:
            query = [list(tup) for tup in query]
            query = [list(tup) for tup in product(*query)]
            query_tensors.append(query)
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
    queries_path = Path(queries_dir).joinpath(f"{workload}.queries.json")

    attributes_domain_sizes = blocks_metadata["attributes_domain_sizes"]
    if workload == "all":
        create_all_queries(
            queries_path, blocks_metadata["attributes_domain_sizes"]
        )  # Query space size for covid dataset: 34425

    # create_specific_queries(queries_path, attributes_domain_sizes)
    # create_specific_queries(queries_path, attributes_domain_sizes, type="2way_marginal_mix")
    else:
        create_specific_queries(queries_path, attributes_domain_sizes, type=workload)


if __name__ == "__main__":
    typer.run(main)
