import socket


class BlocksClient:
    def __init__(self, config) -> None:
        self.config = config
        self.host = self.config.blocks_server.host
        self.port = self.config.blocks_server.port

    def send_request(self, block_data_path):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            s.sendall(block_data_path)
            data = s.recv(1024)

        print(f"Received {data!r}")
