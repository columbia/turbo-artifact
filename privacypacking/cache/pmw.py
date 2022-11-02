import math
from time import sleep

import numpy as np
import torch
from loguru import logger

from privacypacking.budget import BasicBudget
from privacypacking.budget.block import HyperBlock
from privacypacking.budget.curves import BoundedOneShotSVT, GaussianCurve, ZeroCurve
from privacypacking.budget.histogram import DenseHistogram


class PMW:
    def __init__(
        self,
        hyperblock: HyperBlock,
        epsilon=0.1,
        delta=1e-4,
        beta=1e-3,
        k=100,
        accounting="RDP",
    ):
        self.hyperblock = hyperblock
        self.n = hyperblock.size
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.k = k  # max_total_queries
        self.M = hyperblock.domain_size
        self.queries_ran = 0
        self.accounting = accounting
        self.histogram = DenseHistogram(self.M)

        # Initializations as per the Hardt and Rothblum 2010 paper
        self.hard_queries_answered = 0  # `w`
        self.max_hard_queries = self.n * math.sqrt(math.log(self.M))
        self.sigma = (
            10 * math.log(1 / self.delta) * math.pow(math.log(self.M), 1 / 4)
        ) / (math.sqrt(self.n) / self.epsilon)
        self.learning_rate = math.pow(math.log(self.M), 1 / 4) / math.sqrt(self.n)
        self.T = 4 * self.sigma * (math.log(self.k) + math.log(1 / self.beta))

        if self.accounting == "Naive":
            # We consume the whole budget once at initialization (or rather: at the first call)
            self.init_budget = BasicBudget(self.epsilon)
            return

        # Sparse Vector parameters
        self.local_svt_queries_ran = 0
        self.local_svt_max_queries = k
        self.nu = 1
        self.ro = 1
        self.Delta = 1 / self.n  # Query sensitivity

        # The initial threshold should be noisy too if we want to use Sparse Vector
        self.noisy_T = self.T + np.random.normal(0, self.Delta * self.ro)
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
        noisy_error = (
            true_output
            - predicted_output
            + np.random.normal(0, 2 * self.Delta * self.nu)
        )

        # TODO: log error

        if abs(noisy_error) < self.noisy_T:
            # "Output bot" in SVT
            logger.info("easy query")
            sleep(2)
            return predicted_output, run_budget

        # if not self.is_query_hard(noisy_error):
        #     return predicted_output, run_budget

        # NOTE: cut-off = 1 and pay as you go -> no limit on the number of hard queries
        # # Too many hard queries - breaking privacy. Don't update histogram or return query result.
        # if self.hard_queries_answered >= self.max_hard_queries:
        #     # TODO: what do you pay exactly here?
        #     logger.warning("The planner shouldn't let you do this.")
        #     return None, run_budget

        # NOTE: Salil's PMW samples fresh noise here, it makes more sense I think. Clean composition.
        noisy_output = true_output + np.random.normal(0, self.Delta * self.nu)
        run_budget += GaussianCurve(sigma=self.nu)
        noisy_error_2 = noisy_output - predicted_output

        # NOTE: for unlimited kmax, nonnegative queries -> RDP SVT gives a (maybe) tighter theorem. But let's stay "simple" for now.

        # TODO: check how sparse exponential looks like
        self.histogram.multiply(
            torch.exp(-self.learning_rate * np.sign(noisy_error_2) * query)
        )
        self.histogram.normalize()
        self.hard_queries_answered += 1

        # Now, we start a new sparse vector and pay for it
        self.noisy_T = self.T + np.random.normal(0, self.Delta * self.ro)
        run_budget += BoundedOneShotSVT(
            ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
        )
        logger.info("Paying for new sprase vec")
        sleep(2)

        return noisy_output, run_budget


# if __name__ == "__main__":
#     debug1()
