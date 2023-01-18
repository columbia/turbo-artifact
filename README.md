# Precycle

## Organization


## Setup

- Follow these instructions to install Redis https://redis.io/docs/getting-started/installation/install-redis-on-linux/
    Some more help: https://www.tutorialspoint.com/redis/redis_environment.htm

- Run a second Redis instance: https://gist.github.com/Paprikas/ef55f5b2401c4beec75f021590de6a67
    ( We need to different Redis instances: one for the Cache and one for the BlockBudgets)
- Follow these instructions to install Timescaledb/postgres https://docs.timescale.com/install/latest/self-hosted/installation-debian/

- Set up the  database and the hypertable to store covid data
```    CREATE database covid;

    CREATE EXTENSION IF NOT EXISTS timescaledb;


    # Create the covid data table - for simplicity time is an incrementing integer

    CREATE TABLE covid_data (
    time        INT               NOT NULL,
    positive    INT               NOT NULL,
    gender      INT               NOT NULL,
    age         INT               NOT NULL,
    ethnicity   INT               NOT NULL
    );


    # Create the hypertable 'covid-data' and set a chunk-time-interval equal to 1.
    # Now time t takes values 1,2,3,4, and the chunks will be splitted per t -> temporary hack to get easily chunk/block ids.

    SELECT create_hypertable('covid_data', 'time', chunk_time_interval => 1);
```


- Create the default data blocks
```python data/covid19/covid19_data/dataset_generator.py```

- Create the default queries
```python data/covid19/covid19_queries/queries.py```

- Create the default workload
```python data/covid19/covid19_workload/workload_generator.py --requests-type 400:7```


For more options. on creating the data blocks, queries and workload see the relevant read_me files.
## Contributing


- Code style: flake8, Black, Google-style docstrings, type annotations (ideally)

- Install [poetry](https://python-poetry.org/) to set up the environment. The `poetry.lock` file will ensure that we all have the exact same packages. Useful commands:
    + `poetry add plotly` to install a package (e.g. Plotly) and update the requirements
    + `poetry update` to update the dependencies, `poetry lock --no-update` to just refresh the lockfile
    + `poetry install` to install the dependencies
    + `poetry shell` to activate the virtual environment

- Use `nb-clean` to clean heavy notebooks if necessary:
    + `nb-clean clean notebooks/*.ipynb` to run by hand
    + `nb-clean add-filter` to run at each Git commit
    + `nb-clean remove-filter` to switch back to manual cleaning (good if the metadata in some notebooks is actually useful, e.g. logs or graphs)
 
