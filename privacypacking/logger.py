import json


class Logger:
    def __init__(self, file):
        self.file = file

    def log(
        self,
        tasks,
        blocks,
        allocated_task_ids,
        simulator_config,
        compact=False,
        **kwargs
    ):
        with open(self.file, "w") as fp:
            log = {"tasks": []}
            for task in tasks:
                task_dump = task.dump()
                task_dump.update(
                    {"allocated": True if task.id in allocated_task_ids else False}
                )
                log["tasks"].append(
                    task_dump
                )  # todo change allocated_task_ids from list to a set or sth more efficient for lookups

            log["blocks"] = []
            for block in blocks.values():
                log["blocks"].append(block.dump())

            log["simulator_config"] = simulator_config.dump()

            # Any other thing to log
            for key, value in kwargs.items():
                log[key] = value

            if compact:
                json_object = json.dumps(log, separators=(",", ":"))
            else:
                json_object = json.dumps(log, indent=4)

            fp.write(json_object)
