import psycopg2
import pandas as pd
from loguru import logger
from collections import namedtuple
from precycle.budget import SparseHistogram
from precycle.utils.utils import get_blocks_size


class PSQL:
    def __init__(self, config) -> None:
        self.config = config

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
            logger.info(error)
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
            logger.info(error)
        return status

    def run_query(self, query, blocks):
        try:
            cur = self.psql_conn.cursor()
            cur.execute(query)
            true_result = float(cur.fetchone()[0])
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            logger.info(error)
            exit(1)

        blocks_size = get_blocks_size(blocks, self.config.blocks_metadata)
        true_result /= blocks_size
        # print("result:", true_result, "total-size:", blocks_size)
        return true_result

    def close(self):
        if self.psql_conn is not None:
            self.psql_conn.close()
            logger.info("Database connection closed.")


Block = namedtuple("Block", ["size", "histogram"])


class MockPSQL:
    def __init__(self, config) -> None:
        self.config = config

        # Blocks are in-memory histograms
        self.blocks = {}
        self.blocks_count = 0

        self.attributes_domain_sizes = self.config.blocks_metadata[
            "attributes_domain_sizes"
        ]
        self.domain_size = float(self.config.blocks_metadata["domain_size"])

    def add_new_block(self, block_data_path):
        raw_data = pd.read_csv(block_data_path).drop(columns=["time"])
        histogram_data = SparseHistogram.from_dataframe(
            raw_data, self.attributes_domain_sizes
        )
        block_id = self.blocks_count
        block_size = get_blocks_size(block_id, self.config.blocks_metadata)
        block = Block(block_size, histogram_data)
        self.blocks[block_id] = block
        self.blocks_count += 1

    def run_query(self, query, blocks):
        true_result = 0
        blocks_size = 0
        for block_id in range(blocks[0], blocks[1] + 1):
            block = self.blocks[block_id]
            true_result += block.size * block.histogram.run(query)
            blocks_size += block.size
        # print("true result abs", true_result, "block size", blocks_size)
        true_result /= blocks_size
        # print("true result:", true_result, "total-size:", blocks_size)
        return true_result

    def close(self):
        pass
