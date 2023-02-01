Create a new covid19 workload by runnning:

`python3 workload_generator.py --requests-type 30:7 --queries covid19_queries/all.queries.json`

- This workload will create a pool of tasks from the "all.queries" query pool.
- The tasks will request 1, 7, 14, 21 or 28 blocks (step=7, stops at 30).
