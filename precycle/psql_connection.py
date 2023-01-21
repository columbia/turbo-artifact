import psycopg2
import pandas as pd
from collections import namedtuple
from precycle.budget import SparseHistogram
from precycle.sql_converter import SQLConverter
from precycle.tesnor_converter import TensorConverter


class PSQLConnection:
    def __init__(self, config) -> None:
        self.config = config
        self.sql_converter = SQLConverter(config.blocks_metadata)

        # Initialize the PSQL connection
        try:
            # Connect to the PostgreSQL database server
            self.psql_conn = psycopg2.connect(
                host=config.postgres.host,
                database=config.postgres.database,
                user=config.postgres.username,
                password=config.postgres.password,
            )
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(1)

    def add_new_block(self, block_data_path):
        status = b"success"
        try:
            cur = self.psql_conn.cursor()
            cmd = f"""
                    COPY covid_data(time, positive, gender, age, ethnicity)
                    FROM '{block_data_path}'
                    DELIMITER ','
                    CSV HEADER;
                """
            cur.execute(cmd)
            cur.close()
            self.psql_conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            status = b"failed"
            print(error)
        return status

    def run_query(self, query, blocks):
        sql_query = self.sql_converter.query_vector_to_tensor(query, blocks)
        try:
            cur = self.psql_conn.cursor()
            cur.execute(sql_query)
            true_result = float(cur.fetchone()[0])
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(1)

        num_blocks = blocks[1]-blocks[0]+1
        blocks_size = num_blocks * self.config.blocks_metadata['block_size']
        true_result /= blocks_size
        print("result:", true_result, "total-size:", blocks_size)
        return true_result, blocks_size

    def close(self):
        if self.psql_conn is not None:
            self.psql_conn.close()
            print("Database connection closed.")


Block = namedtuple("Block", ["size", "histogram"])


class MockPSQLConnection:
    def __init__(self, config) -> None:
        self.config = config
        self.tensor_convertor = TensorConverter(config.blocks_metadata)

        # Blocks are in-memory histograms
        self.blocks = {}
        self.blocks_count = 0

        self.attributes_domain_sizes = self.config.blocks_metadata[
            "attributes_domain_sizes"
        ]
        self.domain_size = float(self.config.blocks_metadata["domain_size"])

    def add_new_block(self, block_data_path):
        raw_data = pd.read_csv(block_data_path)
        histogram_data = SparseHistogram.from_dataframe(
            raw_data, self.attributes_domain_sizes
        )
        #float(self.config.blocks_metadata["blocks"][str(block_id)]["size"])
        block_size = self.config.blocks_metadata['block_size'] 
        block = Block(block_size, histogram_data)
        self.blocks[self.blocks_count] = block
        self.blocks_count += 1

    def run_query(self, query, blocks):
        tensor_query = self.tensor_convertor.query_vector_to_tensor(query)
        true_result = 0
        blocks_size = 0
        for block_id in range(blocks[0], blocks[1] + 1):
            block = self.blocks[block_id]
            true_result += block.size * block.histogram.run(tensor_query)
            blocks_size += block.size
        true_result /= blocks_size
        print("result:", true_result, "total-size:", blocks_size)
        return true_result, blocks_size

    def close(self):
        pass
