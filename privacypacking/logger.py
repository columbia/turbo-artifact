import json


class Logger:
    def __init__(self, file, scheduler_name):
        self.file = file
        self.scheduler_name = scheduler_name

    def log(self, tasks, blocks, allocated_task_ids):
        with open(self.file, "w") as fp:
            log = {"tasks": []}
            num_scheduled = 0
            for task in tasks:
                task_dump = task.dump()
                if task.id in allocated_task_ids:
                    num_scheduled += 1
                    allocated = True
                else:
                    allocated = False
                task_dump.update(
                    {"allocated": allocated}
                )
                log["tasks"].append(
                    task_dump
                )  # todo change allocated_task_ids from list to a set or sth more efficient for lookups

            log["blocks"] = []
            for block in blocks.values():
                log["blocks"].append(block.dump())

            log.update({"scheduler_name": self.scheduler_name})
            log.update({"num_scheduled_tasks": num_scheduled})
            json_object = json.dumps(log, indent=4)
            fp.write(json_object)
