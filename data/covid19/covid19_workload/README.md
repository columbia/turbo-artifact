Create a new covid19 workload by runnning:
`python workload_generator.py`


- Limiting simulation run to 400 blocks
- A task cannot request for more blocks that those existing
- A task may ask a number of blocks from {1, 7, 14, 30, 60, 90, 120}, latest blocks first        
- continuous aggregate left for later
- Creating only few tasks per day as for now we don't have a lot of queries (2 types)