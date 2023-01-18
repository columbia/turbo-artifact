import socket
import psycopg2


class BlocksServer:
    
    ''' Entrypoint for adding new blocks in Postgres and the 'budget_accountant' KV store. '''

    def __init__(self, psql_conn, budget_accountant, config) -> None:
        self.config = config
        self.host = self.config.host
        self.port = self.config.port
        self.psql_conn = psql_conn
        self.budget_accountant = budget_accountant

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    # Simple blocking connection # TODO: allow for multiple connections
                    data = conn.recv(1024)
                    if not data:
                        break
                    response = self.serve_request(data)
                    conn.sendall(response)

    def serve_request(self, block_data_path):
        # Add the block in the database as a new chunk of data
        print("AA", block_data_path)
        try:
            cur = self.psql_conn.cursor()
            cmd = f"""
                    COPY covid_data(positive, gender, ethnicity, time)
                    FROM '{block_data_path}'
                    DELIMITER ','
                    CSV HEADER;
                """
            cur.execute(cmd)
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            exit(1)

        # Add the block budget in KV store
        status = self.budget_accountant.add_new_block_budget()
        return status
