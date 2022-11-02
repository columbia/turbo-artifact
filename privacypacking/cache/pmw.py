import math
from time import sleep

import numpy as np
import torch
from loguru import logger

from privacypacking.budget import BasicBudget
from privacypacking.budget.block import HyperBlock
from privacypacking.budget.curves import BoundedOneShotSVT, GaussianCurve, ZeroCurve
from privacypacking.budget.histogram import DenseHistogram, flat_items


class PMW:
    def __init__(
        self,
        hyperblock: HyperBlock,
        nu=0.5,  # Scale of noise added on queries
        ro=None,  # Scale of noise added on the threshold. Will be nu if left empty.
        alpha=0.2,  # Max error guarantee (or order of magnitude)
        k=100,
    ):
        # TODO: a friendlier constructor computes nu based on a fraction of block budget (and alpha)

        # Generic PMW arguments
        self.hyperblock = hyperblock
        self.n = hyperblock.size
        self.k = k  # max_total_queries
        self.M = hyperblock.domain_size
        self.queries_ran = 0
        self.histogram = DenseHistogram(self.M)
        self.nu = nu

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
        self.init_budget = BoundedOneShotSVT(
            ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
        )

    def is_query_hard(self, error):
        return abs(error) > self.T

    def run(self, query):
        assert isinstance(query, torch.Tensor)

        # Pay the initialization budget if it's the first call
        if self.init_budget is not None:
            run_budget = self.init_budget
            self.init_budget = None
        else:
            run_budget = ZeroCurve()

        if self.local_svt_queries_ran >= self.local_svt_max_queries:
            logger.warning("Local sparse vector exhausted. Start a new one.")
            # TODO: reinit, pay budget, try again? Or reinit at the previous run?
            return None, run_budget

        # Too many rounds - breaking privacy
        # if self.queries_ran >= self.k:
        #     logger.warning("The planner shouldn't let you do this.")
        #     return None, run_budget

        # true_output = self.hyperblock.run(query)
        # noise_sample = np.random.laplace(scale=self.sigma)
        # noisy_output = true_output + noise_sample

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

        # TODO: log error

        if noisy_error < self.noisy_threshold:
            # "Output bot" in SVT
            logger.info("easy query")
            sleep(2)
            return predicted_output, run_budget

        logger.info(f"Predicted: {predicted_output}, true: {true_output}, hard query")

        # if not self.is_query_hard(noisy_error):
        #     return predicted_output, run_budget

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
        noisy_error_2 = noisy_output - predicted_output

        # NOTE: for unlimited kmax, nonnegative queries -> RDP SVT gives a (maybe) tighter theorem. But let's stay "simple" for now.

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

        # Now, we start a new sparse vector and pay for it
        self.noisy_threshold = self.alpha / 2 + np.random.normal(
            0, self.Delta * self.ro
        )
        run_budget += BoundedOneShotSVT(
            ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
        )
        logger.info("Paying for new sprase vec")
        # sleep(2)

        return noisy_output, run_budget


# if __name__ == "__main__":
#     debug1()
