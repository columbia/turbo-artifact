RENYI_EPSILON = "renyi_epsilon"
RENYI_DELTA = "renyi_delta"
BLOCKS_SPEC = "blocks_spec"
TASKS_SPEC = "tasks_spec"
OFFLINE = "offline"
ONLINE = "online"
ENABLED = "enabled"
NUM = "num"
CURVE_DISTRIBUTIONS = "curve_distributions"
LAPLACE = "laplace"
GAUSSIAN = "gaussian"
SUBSAMPLEGAUSSIAN = "SubsampledGaussian"
NOISE_START = "noise_start"
NOISE_STOP = "noise_stop"
SIGMA_START = "sigma_start"
SIGMA_STOP = "sigma_stop"
DATASET_SIZE = "dataset_size"
BATCH_SIZE = "batch_size"
EPOCHS = "epochs"
SCHEDULER = "scheduler"
SIMPLEX = "simplex"
OFFLINE_DPF = "offline_dpf"

FCFS = "fcfs"
DPF = "dpf"

PLOT_FILE = "plot_file"
FREQUENCY = "frequency"
TASK_ARRIVAL_INTERVAL = "task_arrival_interval"


def update_dict(src, des):
    ref = des
    for k, v in src.items():
        if isinstance(v, dict):
            prev_ref = ref
            ref = ref[k]
            update_dict(v, ref)
            ref = prev_ref
        else:
            ref[k] = v


def get_block_by_block_id(blocks, block_id):
    for block in blocks:
        if block.id == block_id:
            return block
