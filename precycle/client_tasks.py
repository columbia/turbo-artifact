import json
import socket


class TasksClient:
    def __init__(self, config) -> None:
        self.config = config
        self.host = self.config.tasks_server.host
        self.port = self.config.tasks_server.port

    def send_request(self, query_id, query, nblocks, utility, utility_beta):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))

            data = {
                "query_id": query_id,
                "query": query,
                "nblocks": nblocks,
                "utility": utility,
                "utility_beta": utility_beta,
            }
            serialized_data = json.dumps(data).encode('utf-8')
            s.sendall(serialized_data)
            data = s.recv(1024)

        print(f"Received {data!r}")
