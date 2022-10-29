import json
from pathlib import Path
from itertools import chain, combinations, product

# Attributes
attributes = {"positive": 2, "gender": 2, "age": 4, "ethnicity": 8}

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
    path = Path(__file__).resolve().parent.parent.joinpath("covid19_queries/queries.json")
    with open(path, "w") as outfile:
        outfile.write(json_object)



def debug():
    create_queries()

if __name__ == "__main__":
    debug()