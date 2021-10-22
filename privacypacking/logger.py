import json


class Logger:
    def __init__(self, file, scheduler_method):
        self.file = file
        self.scheduler_method = scheduler_method

    # todo: do some housekeeping here

    def get_log_dict(
        self,
        tasks,
        blocks,
        tasks_info,
        allocated_task_ids,
        simulator_config,
        **kwargs,
    ) -> dict:
        log = {"tasks": []}
        num_scheduled = 0
        for task in tasks:
            task_dump = task.dump()
            if task.id in allocated_task_ids:
                num_scheduled += 1
                allocated = True
            else:
                allocated = False
            task_dump.update({"allocated": allocated})
            log["tasks"].append(
                task_dump
            )  # todo change allocated_task_ids from list to a set or sth more efficient for lookups

        log["blocks"] = []
        for block in blocks.values():
            log["blocks"].append(block.dump())

        log["scheduler_method"] = self.scheduler_method
        log["num_scheduled_tasks"] = num_scheduled
        log["total_tasks"] = len(tasks)
        tasks_info = tasks_info.dump()
        # tasks_info["allocated_tasks"]
        log["tasks_scheduling_times"] = sorted(
            tasks_info["tasks_scheduling_time"].values()
        )

        log["simulator_config"] = simulator_config.dump()

        # Any other thing to log
        for key, value in kwargs.items():
            log[key] = value

        return log

    def log(
        self,
        tasks,
        blocks,
        tasks_info,
        allocated_task_ids,
        simulator_config,
        compact=False,
        **kwargs,
    ):
        with open(self.file, "w") as fp:

            log = self.get_log_dict(
                tasks,
                blocks,
                tasks_info,
                allocated_task_ids,
                simulator_config,
                compact=False,
                **kwargs,
            )

            if compact:
                json_object = json.dumps(log, separators=(",", ":"))
            else:
                json_object = json.dumps(log, indent=4)

            fp.write(json_object)
