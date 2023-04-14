from typing import Dict, List, Tuple

import numpy as np

from precycle.utils.utils import (
    HISTOGRAM_RUNTYPE,
    LAPLACE_RUNTYPE,
    LOGS_PATH,
    PMW_RUNTYPE,
)


def compute_hit_scores(
    sv_check_status: List,
    sv_node_id: List,
    laplace_hits: Dict[str, float],
    pmw_hits: Dict[str, float],
    run_types: List,
    budget_per_block: List[Dict],
    node_sizes: List[int],
    total_size: int,
    error: float,
    db_runtimes: List[float],
    true_result_per_node: Dict[str, float],
    runtime: float,
) -> float:
    """
    Given some run metadata, compute how much of the output came from the cache.

    Simple cases:
    - Laplace without VR: hit score = 0 if the query is not present with enough accuracy, hit score = 1 otherwise
    - A single PMW: hit = 0 for hard query, hit = 1 for easy query
    - Aggregation of Laplace only, or PMW only: weighted average of hit scores, with weight = fraction of samples
    - Aggregation of a mix of Laplace and PMW: weighted average too
    - Aggregation of Laplace + one global SV check (on both histograms and Laplace):
        It would be weird to treat it as a single PMW (we can get an easy query on 1 chunk thanks to the 99 Laplace chunks)
        So we treat the global SV as a local SV only on non-Laplace chunks.
    - Laplace with VR: not implemented yet, but we can do something like (new_eps - old_eps) / old_eps in (0,1)


    Laplace-only and SV-only hit score:
    - Weighted Laplace Hits / Laplace_size
    - Weighted SV Hits / SV_size
    - What if SV_size = 0 or Laplace_size = 0? Return NaN and dropNA when computing the hit rate
    - Also return the Histogram/Laplace ratio

    Note:
        - Budget discount = cost_with_cache / cost_without_cache, where cost is an aggregated budget metric (e.g. \sum \eps)
        - Hit score gives both Laplace and SV have the same weight
    """

    # We only consider the first run of each query (in case of SV fail we have to run again)
    run_types = run_types[0]

    laplace_score = sv_score = laplace_size = sv_size = 0
    for node_key, run_type in run_types.items():
        node_size = node_sizes[node_key]

        if run_type == LAPLACE_RUNTYPE:
            laplace_score += node_size * laplace_hits[0][node_key]
            laplace_size += node_size
        elif run_type == HISTOGRAM_RUNTYPE:
            # Hit = 1 if the global SV returns True (easy query), 0 otherwise, for every histogram node
            sv_score += node_size * int(sv_check_status[0])
            sv_size += node_size
        elif run_type == PMW_RUNTYPE:
            # No histograms, only a single PMW or a tree of PMWs. One SV per PMW.
            sv_score += node_size * pmw_hits[0][node_key]
            sv_size += node_size
        else:
            raise NotImplementedError(f"Run type {run_type} not implemented")

    total_score = (laplace_score + sv_score) / total_size
    laplace_score /= laplace_size if laplace_size > 0 else np.NaN
    sv_score /= sv_size if sv_size > 0 else np.NaN

    sv_ratio = sv_size / total_size

    return dict(
        total_hit_score=total_score,
        laplace_hit_score=laplace_score,
        sv_hit_score=sv_score,
        sv_ratio=sv_ratio,
    )
