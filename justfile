# Dumping some useful commands here, in case it helps
# Run with https://github.com/casey/just or copy-paste

run_one:
    python precycle/run_simulation.py --omegaconf precycle/config/precycle_pierre.json

    # python precycle/start_precycle.py --omegaconf precycle/config/precycle.json

run_monoblock_covid19:
    python experiments/runner.cli.caching.py --exp caching_monoblock --dataset covid19 --loguru-level DEBUG

corner_case_pmw: activate
    python precycle/start_precycle.py --omegaconf precycle/config/corner_case_pmw.json

create_queries:
    python data/covid19/covid19_queries/queries.py
    python data/citibike/citibike_queries/queries.py

create_workload:
    python workload_generator.py --queries /citibike_queries/stories.queries.json --workload-dir citibike/citibike_workload/ --blocks-metadata-path citibike/citibike_data/blocks/metadata.json --requests-type "1"

    # Creates covid queries by default
    python workload_generator.py

create_covid_dataset:
    # A bit buggy script? Doesn't write metadata at the right place too 
    # (Kelly: I moved metadata.json in the <blocks> dir on purpose a while back I think. but the script is very messy indeed)
    python data/covid19/covid19_data/dataset_generator.py
    python data/citibike/citibike_data/generate.py

mlflow:
    mlflow ui --backend-store-uri file:///$HOME/precycle/logs/mlruns --port 5003


profile:
    scalene --json --outfile profile.json precycle/run_simulation.py --omegaconf precycle/config/precycle_pierre.json
    # scalene --cli --html --outfile profile.html precycle/run_simulation.py --omegaconf precycle/config/precycle_pierre.json


activate:
    poetry shell