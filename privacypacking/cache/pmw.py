import numpy as np
import torch
from loguru import logger

from privacypacking.budget import Budget
from privacypacking.budget.block import HyperBlock
from privacypacking.budget.curves import BoundedOneShotSVT, GaussianCurve, ZeroCurve
from privacypacking.budget.histogram import DenseHistogram, flat_items
from privacypacking.utils.utils import mlflow_log

class PMW:
    def __init__(
        self,
        hyperblock: HyperBlock,
        nu=485,  # Scale of noise added on queries. 485 comes from epsilon=0.01, delta=1e-5 (yes it's really noisy)
        ro=None,  # Scale of noise added on the threshold. Will be nu if left empty.
        alpha=0.2,  # Max error guarantee (or order of magnitude)
        k=10,  # Max number of queries for each OneShot SVT instance
    ):
        # TODO: some optimizations
        # - a friendlier constructor computes nu based on a fraction of block budget (and alpha)
        # - for unlimited kmax, nonnegative queries -> RDP SVT gives a (maybe) tighter theorem. But let's stay "simple" for now.
        # - cheap version of the previous point: dynamic k, increase over time

        # Generic PMW arguments
        self.hyperblock = hyperblock
        self.n = hyperblock.size
        self.k = k  # max_total_queries
        self.M = hyperblock.domain_size
        self.queries_ran = 0
        self.hard_queries_ran = 0
        self.histogram = DenseHistogram(self.M)
        self.nu = nu
        self.id = str(hyperblock.id)[1:-1].replace(", ", "-")

        # Sparse Vector parameters
        self.alpha = alpha
        self.local_svt_queries_ran = 0
        self.local_svt_max_queries = k
        self.ro = ro if ro else nu
        self.Delta = 1 / self.n  # Query sensitivity

        # The initial threshold should be noisy too if we want to use Sparse Vector
        self.noisy_threshold = self.alpha / 2 + np.random.normal(
            0, self.Delta * self.ro
        )
        # self.init_budget = BoundedOneShotSVT(
        #     ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
        # ) # TODO: will this be used?

    def worst_case_cost(self) -> Budget:
        # Worst case: we need to pay for a new sparse vector (e.g. first query, or first query after cache miss)
        # and we still do a cache miss, so we pay for a true query on top of that
        return BoundedOneShotSVT(
            ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
        ) + GaussianCurve(sigma=self.nu)

    def run(self, query):
        assert isinstance(query, torch.Tensor)

        if self.local_svt_queries_ran >= self.local_svt_max_queries:
            self.local_svt_queries_ran = 0
            logger.warning(
                "Local sparse vector is exhausted (too many easy queries). Starting a new one..."
            )

        # Pay the initialization budget if it's the first call
        if self.local_svt_queries_ran == 0:
            self.noisy_threshold = self.alpha / 2 + np.random.normal(
                0, self.Delta * self.ro
            )
            run_budget = BoundedOneShotSVT(
                ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
            )
        else:
            run_budget = ZeroCurve()

        # Comes for free (public histogram)
        predicted_output = self.histogram.run(query)

        # Never released except in debugging logs
        true_output = self.hyperblock.run(query)
        self.queries_ran += 1
        self.local_svt_queries_ran += 1

        # `noisy_error` is a DP query with sensitivity self.Delta, Sparse Vector needs twice that
        noisy_error = abs(true_output - predicted_output) + np.random.normal(
            0, 2 * self.Delta * self.nu
        )

        if noisy_error < self.noisy_threshold:
            # Easy query, i.e. "Output bot" in SVT
            logger.info("easy query")
            output = predicted_output
        else:
            # Hard query, i.e. "Output top" in SVT
            # We'll start a new sparse vector at the beginning of the next query (and pay for it)
            self.local_svt_queries_ran = 0
            self.hard_queries_ran += 1

            logger.info(
                f"Predicted: {predicted_output}, true: {true_output}, hard query"
            )

            # NOTE: cut-off = 1 and pay as you go -> no limit on the number of hard queries
            # # Too many hard queries - breaking privacy. Don't update histogram or return query result.
            # if self.hard_queries_answered >= self.max_hard_queries:
            #     # TODO: what do you pay exactly here?
            #     logger.warning("The planner shouldn't let you do this.")
            #     return None, run_budget

            # NOTE: Salil's PMW samples fresh noise here, it makes more sense I think.
            #       We could also use yet another noise scaling parameter
            noisy_output = true_output + np.random.normal(0, self.Delta * self.nu)
            run_budget += GaussianCurve(sigma=self.nu)

            # Multiplicative weights update for the relevant bins
            values = query.values()
            if noisy_output > predicted_output:
                # We need to make the estimated count higher to be closer to reality
                updates = torch.exp(values * self.alpha / 8)
            else:
                updates = torch.exp(-values * self.alpha / 8)
            for i, u in zip(query.indices()[1], updates):
                self.histogram.tensor[0, i] *= u
            self.histogram.normalize()
            output = noisy_output

        mlflow_log(f"{self.id}/queries_ran", self.queries_ran, self.queries_ran)
        mlflow_log(f"{self.id}/hard_queries_ran", self.hard_queries_ran, self.queries_ran)
        mlflow_log(f"{self.id}/true_abs_error", abs(predicted_output - true_output), self.queries_ran)
        return output, run_budget