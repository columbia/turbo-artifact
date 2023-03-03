from precycle.budget.histogram import build_sparse_tensor


class QueryConverter:
    def __init__(self, blocks_metadata) -> None:
        self.attribute_names = blocks_metadata["attribute_names"]
        self.attribute_domain_sizes = blocks_metadata["attributes_domain_sizes"]

    def convert_to_sql(self, query_vector, blocks):
        p = [set() for _ in self.attribute_names]

        for entry in query_vector:
            for i, val in enumerate(entry):
                p[i].add(val)

        in_clauses = []
        for i, s in enumerate(p):
            domain_size = self.attribute_domain_sizes[i]
            if len(s) != domain_size:
                if len(s) == 1:
                    in_clauses += [f"{self.attribute_names[i]} = {tuple(s)[0]} "]
                else:
                    in_clauses += [f"{self.attribute_names[i]} IN {tuple(s)} "]
        where_clause = "AND ".join(in_clauses)
        if where_clause:
            where_clause += "AND "
        time_window_clause = f"time>={blocks[0]} AND time<={blocks[1]}"
        sql = (
            f"SELECT COUNT(*) FROM covid_data WHERE {where_clause}{time_window_clause};"
        )
        return sql

    def convert_to_tensor(self, query_vector):
        tensor = build_sparse_tensor(
            bin_indices=query_vector,
            values=[1.0] * len(query_vector),
            attribute_sizes=self.attribute_domain_sizes,
        )
        return tensor
