import json


class SQLConverter:

    def __init__(self, block_metadata_path) -> None:
        with open(block_metadata_path, "r") as f:
            # self.config = config
            # self.database = self.config.postgres.database
            self.metadata = json.load(f)
            self.attribute_names = self.metadata['attribute_names']
            self.attribute_domain_sizes = self.metadata['attributes_domain_sizes']



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
                    where_clause += f'{self.attribute_names[i]} == {x}'
                    if x != len(s)-1:
                        where_clause += ' OR '

                if i != len(p)-1:
                    where_clause += ' AND '
                

        time_window_clause = f"time >= {blocks[0]} AND time <= {blocks[1]}"
        sql = f"SELECT COUNT(*) FROM covid_data WHERE {where_clause}{time_window_clause};"
        return sql



def main():

    query_vector = [
        [
            0,
            0,
            0,
            0
        ],
        [
            0,
            0,
            0,
            1
        ],
        [
            0,
            0,
            0,
            2
        ],
        [
            0,
            0,
            0,
            3
        ],
        [
            0,
            0,
            0,
            4
        ],
        [
            0,
            0,
            0,
            5
        ],
        [
            0,
            0,
            0,
            6
        ],
        [
            0,
            0,
            0,
            7
        ],
        [
            0,
            0,
            1,
            0
        ],
        [
            0,
            0,
            1,
            1
        ],
        [
            0,
            0,
            1,
            2
        ],
        [
            0,
            0,
            1,
            3
        ],
        [
            0,
            0,
            1,
            4
        ],
        [
            0,
            0,
            1,
            5
        ],
        [
            0,
            0,
            1,
            6
        ],
        [
            0,
            0,
            1,
            7
        ],
        [
            0,
            0,
            2,
            0
        ],
        [
            0,
            0,
            2,
            1
        ],
        [
            0,
            0,
            2,
            2
        ],
        [
            0,
            0,
            2,
            3
        ],
        [
            0,
            0,
            2,
            4
        ],
        [
            0,
            0,
            2,
            5
        ],
        [
            0,
            0,
            2,
            6
        ],
        [
            0,
            0,
            2,
            7
        ],
        [
            0,
            0,
            3,
            0
        ],
        [
            0,
            0,
            3,
            1
        ],
        [
            0,
            0,
            3,
            2
        ],
        [
            0,
            0,
            3,
            3
        ],
        [
            0,
            0,
            3,
            4
        ],
        [
            0,
            0,
            3,
            5
        ],
        [
            0,
            0,
            3,
            6
        ],
        [
            0,
            0,
            3,
            7
        ]
    ]

    blocks = (1,10)
    sql_converter = SQLConverter("data/covid19/covid19_data/metadata.json")
    sql = sql_converter.query_vector_to_sql(query_vector, blocks)
    print(sql)


if __name__ == "__main__":
    main()
