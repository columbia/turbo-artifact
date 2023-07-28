# Turbo
Effective caching for linear query workloads over DP databases. Turbo builds upon private multiplicative weights (PMW), a DP mechanism that is powerful in theory but very ineffective in practice, and transforms it into a highly effective caching object, namely PMW-Bypass. A description of this project can be found on our paper, titled [Turbo: Effective caching in Differentially-Private Databases](https://arxiv.org/abs/2306.16163).

## Repo Structure

- `data`: scripts for generating the datasets, queries and workloads for the Covid and Citibike datasets.
- `experiments`: scripts that automate the execution of Turbo experiments concurrently using [Ray](#https://www.ray.io/). You can extend [runner.cli.caching.py](https://github.com/columbia/turbo/blob/artifact/experiments/runner.cli.caching.py) with your own configuration to generate your own experiments.
- `notebooks`: notebooks and utils for visualizing and analyzing execution logs.
- `packaging`: scripts for building Turbo.
- `turbo`: Turbo's core functionality. Refer [here](/turbo/README.md) for Turbo core's structure.


<!-- [For a  guide ](#) -->
  
## Packaging
We package Turbo using the following dockers:
- *Turbo* includes
    - the turbo system
    - the datasets, queries and workloads used in the evaluation for the Covid and Citibike datasets
    - scripts for reproducing all experiments
- *timescaledb* includes an instance of Postgres running the TimescaleDB extension. This is the DBMS used for storing data and running queries.
- *redis-cache* includes an instance of RedisAI for storing differentially-private results and histograms (represented as tensors).
- *redis-budgets* includes an instance of Redis for budget accounting.

## 1. Requirements

Make sure you have a working installation of `docker`.

## 2. Install Turbo
### Download the code

Clone this repository on your machine:
```bash
git clone https://github.com/columbia/turbo.git
```

Enter the repository:
```bash
cd turbo
```

### Build the Turbo docker

Build the docker image for Turbo. This will automatically install all dependencies required for the Turbo system as well as the datasets, queries and workloads used in the evaluation section of the paper. This step takes several minutes to finish (~20') due to the processing and generation of the datasets.
``` bash 
sudo docker build -t turbo -f Dockerfile .
```

## 3. Deploy Postgres and Redis

Deploy Postgres as well as two Redis instances. Postgres is used for storing the datasets and the execution of queries. The first Redis (RedisAI) instance is used for *caching* the differentially-private results or histograms (stored as tensors using Redis-AI). The second Redis instance is used for *budget accounting*.

Check out the default addresses for the three instances in the `docker-compose.yaml` and change them if they are already in use by other services.

``` bash
sudo docker-compose up -d
```
## 4. Reproduce experiments

We prototype Turbo and perform a microbenchmark evaluation using a [simulator](https://github.com/columbia/turbo/tree/artifact/turbo/simulator). This simulates the execution of Turbo by orchestrating the arrival of new queries and data into the system. You can control the simulation and create your own simulation settings by editing the configuration files. The [experiments/runner.cli.caching.py](https://github.com/columbia/turbo/blob/artifact/experiments/runner.cli.caching.py) script automates the execution of multiple experiments concurrently using [Ray](#https://www.ray.io/). You can find the configuration for each experiment hardcoded inside the script.

The script [experiments/ray/run_all.sh](https://github.com/columbia/turbo/blob/artifact/experiments/ray/run_all.sh) contains a complete list of all the commands that generate the experiments presented in the paper. 

### 4.1. Setup
Run `'sudo docker images'` and verify that there are three containers *turbo*, *postgres*, *redis*.

Setup TimescaleDB by creating the databases and hypertables. Use the following command:

``` bash
sudo docker exec -i timescaledb psql -U postgres < packaging/timescaledb.sql
```

If you have changed the default addresses of TimeScaleDB, Redis and RedisAI in the previous steps then update the addresses inside these configuration files, too: `turbo/config/turbo_system_eval_monoblock_covid.json` and `turbo/config/turbo_system_eval_monoblock_citibike.json`. They will be used by experiments that evaluate the system's runtime performance!

### 4.2. Run all experiments


Now, reproduce all Turbo experiments by running the turbo docker with the following command:
``` bash 
sudo docker run -v ~/turbo/logs:/turbo/logs -v ~/turbo/turbo/config:/turbo/turbo/config --network=host --name turbo --shm-size=204.89gb --rm turbo experiments/ray/run_all.sh
```
This step takes around XXX' to finish. 

With the `-v` flag we mount directories `turbo/logs` and `turbo/config` from the host into the container so that the we can access the logs from the host even after the container stops and also allow for the container to access user-defined configurations stored in the host.

### 4.3. Run individual experiments

Alternatively, you can also run experiments from [experiments/ray/run_all.sh](https://github.com/columbia/turbo/blob/artifact/experiments/ray/run_all.sh) individually.
You can enter the Turbo container:
``` bash
sudo docker run -v ~/turbo/logs:/turbo/logs -v ~/turbo/turbo/config:/turbo/turbo/config --network=host --name turbo -it turbo
```
and simply type the python command from [experiments/ray/run_all.sh](https://github.com/columbia/turbo/blob/artifact/experiments/ray/run_all.sh) that corresponds to the experiment you want to reproduce.

For example the following command runs the experiment for Figure 8.a.: 

``` bash
python experiments/runner.cli.caching.py --exp caching_monoblock_heuristics --dataset covid19
```

If you want to bypass `runner.cli.caching.py` you can directly call the simulation entrypoint with your configuration file as an argument.

For example the following command runs the experiment for Figure 10.d.: 
``` bash
python run_simulation.py --omegaconf turbo/config/turbo_system_eval_monoblock_covid.json
```

Note that, for system's performance evaluation we bypass `runner.cli.caching.py` since we do not wish to parallelize the experiments.

`run_simulation` is the official entrypoing for running microbenchmarks under controled simulation settings.


##  4.4. Analyze results
The [experiments/runner.cli.caching.py](https://github.com/columbia/turbo/blob/artifact/experiments/runner.cli.caching.py) script will automatically analyze the execution logs and create plots corresponding to the figures presented in the paper. 
Note that the terminology used in the labels might differ slightly from that used in the paper. 
Typically, you will see that here we refer to `Turbo` as `Hybrid` which is an alternative name we use internally to describe Turbo.

# 5. Interactive use of Turbo

