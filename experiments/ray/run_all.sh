#!/bin/sh

echo "Running Figures 7.a and 7.b (static-non-partitioned covid zipfs 0/1)"
python experiments/runner.cli.caching.py --exp caching_monoblock --dataset covid19

echo "Running Figure 7.c and 7.b (static-non-partitioned citibike zipf 0)"
python experiments/runner.cli.caching.py --exp caching_monoblock --dataset citibike

echo "Running Figure 7.d (static-non-partitioned covid empirical convergence)"
python experiments/runner.cli.caching.py --exp convergence --dataset covid19

echo "Running Figure 8.a (static-non-partitioned covid heuristics)"
python experiments/runner.cli.caching.py --exp caching_monoblock_heuristics --dataset covid19

echo "Running Figure 8.b (static-non-partitioned covid learning rate)"
python experiments/runner.cli.caching.py --exp caching_monoblock_learning_rates --dataset covid19

echo "Running Figure 9.a and 9.b (static-partitioned covid zipfs 0/1)"
python experiments/runner.cli.caching.py --exp caching_static_multiblock_laplace_vs_hybrid --dataset covid19

echo "Running Figure 9.c (static-partitioned citibike zipf 0)"
python experiments/runner.cli.caching.py --exp caching_static_multiblock_laplace_vs_hybrid --dataset citibike

echo "Running Figure 10.a and 10.b (streaming-partitioned covid zipfs 0/1)"
python experiments/runner.cli.caching.py --exp caching_streaming_multiblock_laplace_vs_hybrid --dataset covid19

echo "Running Figure 10.c (streaming-partitioned citibike zipf 0)"
python experiments/runner.cli.caching.py --exp caching_streaming_multiblock_laplace_vs_hybrid --dataset citibike