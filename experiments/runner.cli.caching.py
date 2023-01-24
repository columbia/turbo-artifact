import typer

from experiments.ray_runner import grid_online

app = typer.Typer()


def caching():

    task_paths = ["1:1blocks_10queries.privacy_tasks.csv"]

    grid_online(
        logs_dir="experiment",
        tasks_path=task_paths,
        blocks_path="covid19/covid19_data/blocks",
        blocks_metadata="covid19/covid19_data/metadata.json",
        planner=["MinCutsPlanner"],
        cache=["DeterministicCache"],  # ProbabilisticCache
        initial_blocks=[1],
        max_blocks=[0],
        avg_num_tasks_per_block=[5e4],
        max_tasks=[5e4],
        initial_tasks=[0],
        enable_random_seed=True,
        alpha=[0.005],
        beta=[0.0001],
    )


@app.command()
def run(exp: str = "caching"):
    globals()[f"{exp}"]()


if __name__ == "__main__":
    app()
