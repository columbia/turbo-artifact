# Dumping some useful commands here, in case it helps
# Run with https://github.com/casey/just or copy-paste

run_one:
    python precycle/run_simulation.py --omegaconf precycle/config/precycle_pierre.json

run_monoblock_covid19:
    python experiments/runner.cli.caching.py --exp caching_monoblock --dataset covid19 --loguru-level DEBUG

corner_case_pmw: activate
    python precycle/start_precycle.py --omegaconf precycle/config/corner_case_pmw.json

create_queries: create_covid_queries create_citibike_queries

create_covid_queries:
    python data/covid19/covid19_queries/queries.py

create_citibike_queries:
    python data/citibike/citibike_queries/queries.py

create_workload: create_citibike_workload create_covid_workload

create_citibike_workload:
    python data/workload_generator.py --queries data/citibike/citibike_queries/stories.queries.json --workload-dir data/citibike/citibike_workload/ --blocks-metadata-path data/citibike/citibike_data/blocks/metadata.json --requests-type "1"

create_covid_workload:
    python data/workload_generator.py

create_covid_dataset:
    cd data/covid19/covid19_data ; python dataset_generator.py
    
create_citibike_dataset:
    python data/citibike/citibike_data/generate.py

mlflow:
    mlflow ui --backend-store-uri file:///$HOME/precycle/logs/mlruns --port 5003

profile:
    scalene --json --outfile profile.json precycle/run_simulation.py --omegaconf precycle/config/precycle_pierre.json
    # scalene --cli --html --outfile profile.html precycle/run_simulation.py --omegaconf precycle/config/precycle_pierre.json

run:
    #!/usr/bin/env python
    print("Hello")
    def main():
        print("World")
    main()

activate:
    poetry shell