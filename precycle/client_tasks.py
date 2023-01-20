import json
import socket

class TasksClient:
    def __init__(self, config) -> None:
        self.config = config
        self.host = self.config.host
        self.port = self.config.port

    def send_request(self, task):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            serialized_data = json.dumps(task).encode("utf-8")
            s.sendall(serialized_data)
            data = s.recv(4096)
        print(f"Received {data!r}")