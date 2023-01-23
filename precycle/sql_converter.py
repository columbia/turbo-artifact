class SQLConverter:
    def __init__(self, blocks_metadata) -> None:
        self.attribute_names = blocks_metadata["attribute_names"]
        self.attribute_domain_sizes = blocks_metadata["attributes_domain_sizes"]

    def query_vector_to_sql(self, query_vector, blocks):
        p = [set() for _ in self.attribute_names]

        for entry in query_vector:
            for i, val in enumerate(entry):
                p[i].add(val)

        where_clause = ""
        for i, s in enumerate(p):
            domain_size = self.attribute_domain_sizes[i]
            if len(s) != domain_size:
                for x in range(len(s)):
                    where_clause += f"{self.attribute_names[i]}={x}"
                    if x != len(s) - 1:
                        where_clause += " OR "

                if i != len(p) - 1:
                    where_clause += " AND "

        time_window_clause = f"time>={blocks[0]} AND time<={blocks[1]}"
        sql = (
            f"SELECT COUNT(*) FROM covid_data WHERE {where_clause}{time_window_clause};"
        )
        return sql
