import json

from loguru import logger

from privacypacking.schedulers.utils import ALLOCATED


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

        # TODO: remove allocating_task_id from args

        log = {"tasks": []}
        num_scheduled = 0
        # info = tasks_info.dump()
        tasks_scheduling_times = []
        allocated_tasks_scheduling_delays = []

        for task in tasks:
            task_dump = task.dump()

            if tasks_info.tasks_status[task.id] == ALLOCATED:
                num_scheduled += 1
                allocated = True
                tasks_scheduling_times.append(tasks_info.scheduling_time[task.id])
                allocated_tasks_scheduling_delays.append(
                    tasks_info.scheduling_delay.get(task.id, None)
                )
            else:
                allocated = False

            task_dump.update(
                {
                    # "allocated": allocated,
                    "allocated": tasks_info.tasks_status[task.id] == ALLOCATED,
                    "status": tasks_info.tasks_status[task.id],
                    "creation_time": tasks_info.creation_time[task.id],
                    "scheduling_time": tasks_info.scheduling_time.get(task.id, None),
                    "scheduling_delay": tasks_info.scheduling_delay.get(task.id, None),
                    "allocation_index": tasks_info.allocation_index.get(task.id, None),
                }
            )
            log["tasks"].append(
                task_dump
            )  # todo change allocated_task_ids from list to a set or sth more efficient for lookups

        # TODO: Store scheduling times into the tasks directly?

        log["blocks"] = []
        for block in blocks.values():
            log["blocks"].append(block.dump())

        log["scheduler_method"] = self.scheduler_method
        log["num_scheduled_tasks"] = num_scheduled
        log["total_tasks"] = len(tasks)
        log["tasks_info"] = tasks_info.dump()

        # tasks_info = tasks_info.dump()
        # # tasks_info["allocated_tasks"]
        log["tasks_scheduling_times"] = sorted(tasks_scheduling_times)
        log["allocated_tasks_scheduling_delays"] = allocated_tasks_scheduling_delays

        log["simulator_config"] = simulator_config.dump()

        # Any other thing to log
        for key, value in kwargs.items():
            log[key] = value

        return log

    def save_logs(self, log_dict, compact=False, compressed=False):
        if compressed:
            raise NotImplementedError
        else:
            with open(self.file, "w") as fp:
                if compact:
                    json_object = json.dumps(log_dict, separators=(",", ":"))
                else:
                    json_object = json.dumps(log_dict, indent=4)

                fp.write(json_object)

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
        logger.info(f"Saved logs to {self.file}")
