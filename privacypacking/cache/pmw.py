import math
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from termcolor import colored

from privacypacking.budget import BasicBudget
from privacypacking.budget.block import Block
from privacypacking.cache import cache
from privacypacking.budget.histogram import DenseHistogram
from privacypacking.cache.utils import get_splits


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

    def run_cache(self, query_tensor):

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
        true_output = self.dataset.run_query(query_tensor)
        noise_sample = np.random.laplace(scale=self.sigma)
        noisy_output = true_output + noise_sample
        predicted_output = self.histogram.run_query(query_tensor)

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



def debug1():
    pass
    # # Testing ...
    # num_features = 2
    # domain_size_per_feature = {"new_cases": 2, "new_deaths": 2}

    # domain_size = 1
    # for v in domain_size_per_feature.values():
    #     domain_size *= v

    # histogram = Histogram(num_features, domain_size_per_feature, domain_size)

    # histogram.update_bins([(1, 0), (1, 1)], [1, 2])
    # histogram.update_bins([(0, 1), (1, 1)], [3, 4])
    # bins = histogram.get_bins(0)
    # bins = histogram.get_bins(1)
    # histogram.run_task(0)

    # self.n = 100         # block size
    # self.epsilon = 0.1
    # self.delta = 0.01
    # self.beta = 0.001
    # self.M = domain_size
    # self.k = 100


if __name__ == "__main__":
    debug1()
