# Turbo

`Effective caching in differentially-private databases`

# Repo Structure

- `data`: scripts for generating the datasets, queries and workloads for the Covid and Citibike datasets.
- `experiments`: scripts that automate the execution of Turbo experiments concurrently using [Ray](#https://www.ray.io/). You can extend [runner.cli.caching.py](https://github.com/columbia/turbo/blob/artifact/experiments/runner.cli.caching.py) with your own configuration to generate your own experiments.
- `notebooks`: notebooks and utils for visualizing and analyzing execution logs.
- `packaging`: scripts for building Turbo
- `turbo`: Turbo's core functionality. Refer [here](/turbo/README.md) for Turbo core's structure.


<!-- [For a  guide ](#) -->
  
We package Turbo using the following dockers
- *Turbo* container includes
    - the turbo system
    - the datasets, queries and workloads used in the evaluation for the Covid and Citibike datasets
    - scripts for reproducing all experiments

- *timescaledb* container includes an instance of Postgres running the TimescaleDB extension. This is the DBMS used for storing data and running queries.

- *redis-cache* container for storing differentially-private results and histograms
- *redis-budgets* container for budget accounting

# 1. Requirements

Make sure you have a working installation of `docker`.

# 2. Install Turbo
## Download the code

Clone this repository on your machine:

```bash
git clone https://github.com/columbia/turbo.git
```

Enter the repository:

```bash
cd turbo
```

## Build the Turbo docker

Build the docker image for Turbo. This will automatically install all dependencies required for the Turbo system as well as the datasets, queries and workloads used in the evaluation section of the paper. This step takes several minutes to finish (~20') due to the processing and generation of the datasets.

``` bash 
sudo docker build -t turbo -f Dockerfile .
```

# 3. Deploy Postgres and Redis

Deploy Postgres as well as two Redis instances. Postgres is used for storing the datasets and the execution of queries. The first Redis (RedisAI) instance is used for *caching* the differentially-private results or histograms (stored as tensors using Redis-AI). The second Redis instance is used for *budget accounting*.

Check out the default addresses for the three instances in the `docker-compose.yaml` and change them if they are already in use by other services.

``` bash
sudo docker-compose up -d
```
# 4. Reproduce experiments


Run `'sudo docker images'` and verify that there are three containers *turbo*, *postgres*, *redis*.

Setup TimescaleDB by creating the databases and hypertables. Use the following script:

``` bash
sudo docker exec -i timescaledb psql -U postgres < packaging/timescaledb.sql
```


If you have changed the default addresses of TimeScaleDB, Redis and RedisAI in the previous steps then update the addresses inside these configuration files, too: `turbo/config/turbo_system_eval_monoblock_covid.json` and `turbo/config/turbo_system_eval_monoblock_citibike.json`. They will be used by experiments that evaluate the system runtime performance!

Now, reproduce all Turbo experiments by running the turbo docker with the following command:
This step takes around XXX' to finish.

``` bash 
sudo docker run -v ~/turbo/logs:/turbo/logs -v ~/turbo/turbo/config:/turbo/turbo/config -v ~/turbo/turbo/data:/turbo/turbo/data --network=host --name turbo --shm-size=204.89gb --rm turbo experiments/ray/run_all.sh
```

sudo docker run -v ~/turbo/logs:/turbo/logs -v ~/turbo/turbo/config:/turbo/turbo/config --name turbo --shm-size=204.89gb --rm turbo `chmod 777 turbo/run_simulation.py && /bin/bash python turbo/run_simulation.py --omegaconf turbo/config/turbo_system_eval_monoblock_covid.json`

With the `-v` flag we mount directories `turbo/logs` and `turbo/config` from the host into the container so that the we can access the logs from the host even after the container stops and also allow for the container to access user-defined configurations stored in the host.

Here is a useful command for simply entering the docker:
``` bash
sudo docker run -v ~/turbo/logs:/turbo/logs -v ~/turbo/turbo/config:/turbo/turbo/config --network=host --name turbo -it turbo
```



