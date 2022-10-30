import math
import numpy as np
import torch
from privacypacking.budget import BasicBudget
from privacypacking.budget.block import Block
from privacypacking.budget.histogram import DenseHistogram
from privacypacking.budget.queries import Tensor


# TODO: generalize to list of blocks
class PMW:
    def __init__(
        self,
        block: Block,
        epsilon=0.1,
        delta=1e-4,
        beta=1e-3,
        k=100,
    ):
        self.dataset = block.histogram_data     # for more blocks aggregate multiple histograms
        self.block = block
        self.n = len(block)
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.k = k
        self.M = block.histogram_data.domain_size
        self.queries_ran = 0

        # Initializations as per the Hardt and Rothblum 2010 paper
        self.w = 0
        self.sigma = (
            10 * math.log(1 / self.delta) * math.pow(math.log(self.M), 1 / 4)
        ) / (math.sqrt(self.n) / self.epsilon)
        self.learning_rate = math.pow(math.log(self.M), 1 / 4) / math.sqrt(self.n)
        self.T = 4 * self.sigma * (math.log(self.k) + math.log(1 / self.beta))

        self.histogram = DenseHistogram(self.M)

        # We consume the whole budget once at initialization
        self.init_budget = BasicBudget(self.epsilon)

    def is_query_hard(self, error):
        return abs(error) > self.T

    def run(self, query):
        assert isinstance(query, torch.Tensor)

        # We consume the whole budget once at initialization (or rather: at the first call)
        run_budget = None
        if self.init_budget is not None:
            run_budget = self.init_budget
            self.init_budget = None

        # Too many rounds - breaking privacy
        if self.queries_ran >= self.k:
            # TODO: proper error handling/return values
            exit(0)

        self.queries_ran += 1

        # TODO: use efficient dot product here (blocks as sparse histograms)
        # true_output = self.dataset.run_query(query_tensor)
        true_output = self.block.run(query)
        noise_sample = np.random.laplace(scale=self.sigma)
        noisy_output = true_output + noise_sample
        predicted_output = self.histogram.run(query)

        # TODO: log error
        noisy_error = noisy_output - predicted_output

        if not self.is_query_hard(noisy_error):
            return predicted_output, run_budget

        # NOTE: we need to increment w before checking, not after!
        self.w += 1
        # Too many hard queries - breaking privacy
        if self.w > self.n * math.sqrt(math.log(self.M)):
            # Also: I exit before updating the histogram, to be on the safe side
            # (so we won't leak extra privacy if we decide to reuse the histogram without accuracy guarantees later)
            exit(0)

        # TODO: check how sparse exponential looks like
        self.histogram.multiply(
            torch.exp(-self.learning_rate * np.sign(noisy_error) * query_tensor)
        )
        self.histogram.normalize()

        return true_output, run_budget


# if __name__ == "__main__":
#     debug1()
