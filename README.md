# Precycle

## Organization

`data/covid` : scripts for creating the dataset, queries and workload
`precycle/budget`: various implementations for the privacy budget
`precycle/task.py`: the Task specification

`precycle/cache`: deterministic / probabilistic caches for storing results. Implemented as Redis key-value stores. There are 'Mock' versions for both with simple in-memory dictionaries for key-value stores.
`precycle/planner`: max_cuts / min_cuts / ILP - our three versions of the planner
`precycle/budget_accountant.py`: Redis key-value store for storing the budgets of blocks. There is a 'Mock' versions with a simple in-memory dictionary for the key-value store.
`precycle/psql`: An API for storing block data to TimeScaleDB using and running SQL queries using a PSQL client.  There is a 'Mock' version that stores block data in a in-memory dictionary as "histograms" instead, and runs queries using tensor operations.

`precycle/server_blocks.py`: a Blocks Server listening to a socket for new requests for adding block data (both in TimeScaleDB and in the budget_accountant).
`precycle/server_tasks.py`: a Tasks Server listening to a socket for new requests for running a query.
`precycle/client_blocks.py`: API for sending requests to the Blocks Server
`precycle/server_blocks.py`: API for sending requests to the Tasks Server
`precycle/db-functions.py`: If not using the mock versions of the above modules, this file contains functions that help checking/modifying the state of them (TimeScaleDB, Redis instances)

`precycle/query_processor.py`: Finds a DP plan for the query if possible, runs it, consumes budget if necessary and stores metadata.
`precycle/executor.py`: executes the DP plan of a query using the caches and the PSQL module.

`precycle/simulator`: a simulation of the execution of precycle implemented using Simpy. It generates a workload and data blocks given the configuration in `precycle/precycle.json`. It bypasses the blocks/tasks servers API and directly uses the rest of the package modules to execute queries and store data blocks and block budgets.
`precycle/run_simulation.py`: Entrypoint for running precycle in a simulation.

`precycle/precycle.json`: configuration file to setup the precycle execution. Contains configuration for the simulation as well. 
To run everything using the mock modules set the flag "mock": true.


## Setup

If you run without using the mock modules you need to setup Postgres, TimeScaleDB and two Redis instances following the steps below:

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

- You can use the db-functions.py script to manually check or change the status of the databases
    Examples:
    ```python3 precycle/db-functions.py --storage postgres --function get-all```
    ```python3 precycle/db-functions.py --storage redis-budgets --function get-all```
    ```python3 precycle/db-functions.py --storage redis-cache --function get-all```
    ```python3 precycle/db-functions.py --storage postgres --function get-all```


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
 
