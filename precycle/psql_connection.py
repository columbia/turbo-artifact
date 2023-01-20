import typer
import psycopg2
from omegaconf import OmegaConf
from precycle.utils.utils import DEFAULT_CONFIG_FILE

app = typer.Typer()


class PSQLConnection:
    def __init__(self, config, sql_converter) -> None:
        self.config = config
        self.sql_converter = sql_converter

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
        sql_query = self.sql_converter.query_vector_to_sql(query, blocks)
        try:
            cur = self.psql_conn.cursor()
            cur.execute(sql_query)
            true_result = float(cur.fetchone()[0])
            print(true_result)
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(1)

        return true_result
        

    def close(self):
        if self.psql_conn is not None:
            self.psql_conn.close()
            print("Database connection closed.")


@app.command()
def run(
    omegaconf: str = "precycle/config/precycle.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)



if __name__ == "__main__":
    app()
