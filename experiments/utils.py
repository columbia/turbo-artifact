from turbo.utils.utils import REPO_ROOT


def get_paths(dataset):
    tasks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_workload/")
    blocks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_data/")
    blocks_metadata = str(blocks_path_prefix.joinpath("blocks/metadata.json"))
    blocks_path = str(blocks_path_prefix.joinpath("blocks"))
    return blocks_path, blocks_metadata, tasks_path_prefix
