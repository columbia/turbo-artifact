# Turbo

`Effective caching in differentially-private databases`

# Repo Structure

- `data`: scripts for generating the datasets, queries and workloads for the Covid and Citibike datasets.
- `experiments`: scripts that automate the execution of Turbo experiments concurrently using [Ray](#https://www.ray.io/). You can extending [runner.cli.caching.py](https://github.com/columbia/turbo/blob/artifact/experiments/runner.cli.caching.py) with your own configuration to generate your own experiments.
- `notebooks`: notebooks and utils for visualizing experiments' logs.
- `packaging`: scripts for building Turbo
- `turbo`: Turbo's core functionality. Refer [here](/turbo/README.md) for Turbo core's structure.


<!-- [For a  guide ](#) -->
  
We package Turbo using the following dockers
- *Turbo* container includes
    - the turbo system
    - the datasets, queries and workloads used in the evaluation for the Covid and Citibike datasets
    - scripts for reproducing all experiments
    - two Redis instances for Caching and Budget accounting

- *TimescaleDB* container includes an instance of Postgres running the TimescaleDB extension. This is the DBMS used for storing data and running queries.

# 1. Requirements

Make sure you have a working installation of `docker` and `docker-compose`.

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

Deploy Postgres as well as two Redis instances. Postgres is used for storing the datasets and the execution of queries. The first Redis instance is used for *caching* the differentially-private results or histograms (as tensors using Redis-AI). The second Redis instance is used for *budget accounting*.
``` bash
sudo docker-compose up -d --build
```
# 4. Reproduce experiments

Run `'sudo docker images'` and verify that there are three containers *turbo*, *postgres*, *redis*.

Now, run the turbo docker using the following command:
``` bash 
sudo docker run -v ~/turbo/logs:/turbo/logs  --name turbo --shm-size=204.89gb --rm turbo experiments/ray/run_all.sh
```
sudo docker run --name turbo --shm-size=204.89gb -it --rm turbo
This will run all the experiments found in the evaluation section of the paper.
This step takes around XXX' to finish.





