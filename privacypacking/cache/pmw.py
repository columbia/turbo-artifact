import math

import numpy as np
import torch
from loguru import logger

from privacypacking.budget import BasicBudget
from privacypacking.budget.block import HyperBlock
from privacypacking.budget.histogram import DenseHistogram


class PMW:
    def __init__(
        self,
        hyperblock: HyperBlock,
        epsilon=0.1,
        delta=1e-4,
        beta=1e-3,
        k=100,
        accounting="Naive",
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

        # Initializations as per the Hardt and Rothblum 2010 paper
        self.hard_queries_answered = 0  # `w`
        self.max_hard_queries = self.n * math.sqrt(math.log(self.M))
        self.sigma = (
            10 * math.log(1 / self.delta) * math.pow(math.log(self.M), 1 / 4)
        ) / (math.sqrt(self.n) / self.epsilon)
        self.learning_rate = math.pow(math.log(self.M), 1 / 4) / math.sqrt(self.n)
        self.T = 4 * self.sigma * (math.log(self.k) + math.log(1 / self.beta))

        self.histogram = DenseHistogram(self.M)

        # We consume the whole budget once at initialization (or rather: at the first call)
        self.init_budget = (
            BasicBudget(self.epsilon) if self.accounting == "Naive" else None
        )

    def is_query_hard(self, error):
        return abs(error) > self.T

    def run(self, query):
        assert isinstance(query, torch.Tensor)

        # Pay the initialization budget if it's the first call
        run_budget = None
        if self.init_budget is not None:
            run_budget = self.init_budget
            self.init_budget = None

        # Too many rounds - breaking privacy
        if self.queries_ran >= self.k:
            logger.warning("The planner shouldn't let you do this.")
            return None, run_budget

        true_output = self.hyperblock.run(query)
        noise_sample = np.random.laplace(scale=self.sigma)
        noisy_output = true_output + noise_sample
        self.queries_ran += 1

        predicted_output = self.histogram.run(query)

        # TODO: log error
        noisy_error = noisy_output - predicted_output

        if not self.is_query_hard(noisy_error):
            return predicted_output, run_budget

        # Too many hard queries - breaking privacy. Don't update histogram or return query result.
        if self.hard_queries_answered >= self.max_hard_queries:
            # TODO: what do you pay exactly here?
            logger.warning("The planner shouldn't let you do this.")
            return None, run_budget

        # TODO: Salil's survey samples fresh noise here, it makes more sense I think

        # TODO: check how sparse exponential looks like
        self.histogram.multiply(
            torch.exp(-self.learning_rate * np.sign(noisy_error) * query)
        )
        self.histogram.normalize()
        self.hard_queries_answered += 1

        return noisy_output, run_budget


# if __name__ == "__main__":
#     debug1()
