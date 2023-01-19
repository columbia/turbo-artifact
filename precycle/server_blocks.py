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
            print(f"Connected by {addr}")
            with conn:
                while True:
                    try:
                        # Simple blocking connection # TODO: allow for multiple connections
                        data = conn.recv(1024)
                        if not data:
                            continue
                        response = self.serve_request(data.decode())
                        conn.sendall(response)
                    except (Exception) as error:
                        print(error)    
                        exit(1)



    def serve_request(self, block_data_path):
        # # Add the block in the database as a new chunk of data
        status = b"success"
        try:
        #     cur = self.psql_conn.cursor()
        #     cmd = f"""
        #             COPY covid_data(time, positive, gender, age, ethnicity)
        #             FROM '{block_data_path}'
        #             DELIMITER ','
        #             CSV HEADER;
        #         """
        #     cur.execute(cmd)
        #     cur.close()
            
        #     self.psql_conn.commit()

            # TODO: if commit succeeds redis shouldn't fail
            # Add the block budget in KV store
            self.budget_accountant.add_new_block_budget()

        except (Exception, psycopg2.DatabaseError) as error:
            status = b"failed"
            print(error)

        return status
