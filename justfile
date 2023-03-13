# Dumping some useful commands here, in case it helps
# Run with https://github.com/casey/just or copy-paste

run_one:
    python precycle/run_simulation.py --omegaconf precycle/config/precycle_pierre.json

    # python precycle/start_precycle.py --omegaconf precycle/config/precycle.json

run_monoblock:
    python experiments/runner.cli.caching.py --exp caching_monoblock --loguru-level DEBUG

corner_case_pmw: activate
    python precycle/start_precycle.py --omegaconf precycle/config/corner_case_pmw.json

create_queries:
    python data/covid19/covid19_queries/queries.py

create_workload:
    python data/covid19/covid19_workload/workload_generator.py --requests-type 1:2:4 --queries covid19_queries/all.queries.json

create_covid_dataset:
    # A bit buggy script? Doesn't write metadata at the right place too
    python data/covid19/covid19_data/dataset_generator.py

activate:
    poetry shell