# Turbo

Reproducing `Turbo: Effective caching in differentially-private databases`


# 1. Reproduce experiments

This section explains how to reproduce the entire evaluation of Turbo. 
For the artifact evaluation we create the following dockers:
- *Turbo* container includes
    - the turbo system
    - the datasets, queries and workloads used in the evaluation for the Covid and Citibike datasets
    - scripts for reproducing all experiments
    - two Redis instances for Caching and Budget accounting

- *TimescaleDB* container includes an instance of Postgres running the TimescaleDB extension. This is the DBMS used for storing data and running queries.


## 1.1 Requirements

Make sure you have a working installation of `docker` and `docker-compose`.

