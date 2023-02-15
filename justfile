# Dumping some useful commands here, in case it helps
# Run with https://github.com/casey/just or copy-paste

run_one:
    python precycle/start_precycle.py --omegaconf precycle/config/precycle.json

run_experiment:
    echo "TODO"

corner_case_pmw: activate
    python precycle/start_precycle.py --omegaconf precycle/config/corner_case_pmw.json

create_queries:
    python data/covid19/covid19_queries/queries.py

create_covid_dataset:
    # A bit buggy script? Doesn't write metadata at the write place too
    python data/covid19/covid19_data/dataset_generator.py

activate:
    poetry shell