import json
import socket
from loguru import logger
from precycle.task import Task


class TasksServer:

    """Entrypoint for sending new user requests/tasks to the system."""

    def __init__(self, query_processor, budget_accountant, config) -> None:
        self.config = config
        self.host = self.config.host
        self.port = self.config.port

        self.tasks_count = 0

        self.query_processor = query_processor
        self.budget_accountant = budget_accountant

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    logger.info(f"Connected by {addr}")
                    # Simple blocking connection # TODO: allow for multiple connections
                    data = conn.recv(4096)
                    if not data:
                        continue
                    deserialized_data = json.loads(data).decode("utf-8")
                    print(deserialized_data)
                    response = self.serve_request(deserialized_data)
                    conn.sendall(response)  # Send response

    def serve_request(self, data):
        task_id = self.tasks_count
        self.tasks_count += 1

        num_requested_blocks = int(data["nblocks"])
        num_blocks = self.budget_accountant.get_blocks_count()

        if num_requested_blocks > num_blocks:
            logger.info("return an error")
            return

        # Latest Blocks first
        requested_blocks = (num_blocks - num_requested_blocks, num_blocks - 1)

        # At this point user's request should be translated to a collection of block/chunk ids
        task = Task(
            id=task_id,
            query_id=int(data["query_id"]),
            query_type="linear",
            query=eval(data["query"]),
            blocks=requested_blocks,
            n_blocks=num_requested_blocks,
            utility=float(data["utility"]),
            utility_beta=float(data["utility_beta"]),
            name=task_id,
        )

        run_metadata = self.query_processor.try_run_task(task)
        return run_metadata
