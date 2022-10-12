from queue import Empty
from prometheus_client import Histogram

from tomlkit import value
from privacypacking.cache.utils import get_splits
from termcolor import colored
from privacypacking.cache import cache
import math
import numpy as np
import pandas as pd
import yaml


class histogram:
    def __init__(self, num_features, domain_size_per_feature, domain_size) -> None:
        self.domain_size_per_feature = domain_size_per_feature
        self.num_features = num_features
        self.domain_size = domain_size

        normalization_factor = 1 / self.domain_size
        shape = tuple([self.domain_size_per_feature[i] for i in range(num_features)])
        # Initializing with a uniform distribution
        self.bins = np.full(shape=shape, value=normalization_factor)

    def get_bins(
        self, query_id
    ):  # Gives us direct access to the bins we are interested in
        # Two types of hardcoded queries for now: count new cases, count new deaths
        # histogram arrangement:
        # [[new_cases_0, new_deaths_0],
        #  [new_cases_1, new_deaths_1]]
        # todo: generalize

        if query_id == 0:  # count new cases
            return self.bins[1, 0]
        if query_id == 0:  # count new deaths
            return self.bins[1, 1]

    def update_bins(self, indices, value):
        bins = self.bins
        for idx in indices:
            bins = bins[idx]
        bins = value

    def run_task(self, query_id):
        return np.sum(self.get_bins(query_id))

    def dump(
        self,
    ):
        return self.bins


# Todo: make this inherit from a Cache class
# One PMW per Block
class PerBlockPMW:
    def __init__(self, scheduler):

        self.scheduler = scheduler

        #############################
        # todo: for now all this is hardcoded
        num_features = 2
        domain_size = 4
        domain_size_per_feature = {"new_cases": 2, "new_deaths": 2}
        self.n = 100  # block size
        self.epsilon = 0.1
        self.delta = 0.01
        self.beta = 0.001
        self.M = domain_size
        self.k = 100
        #############################

        # Initializations as per the Hardt and Rothblum 2010 paper
        self.sigma = (
            (10 * math.log(1 / self.delta) * math.pow(math.log(self.M), 1 / 4))
            / math.sqrt(self.n)
            * self.epsilon
        )
        self.learning_rate = math.pow(math.log(self.M), 1 / 4) / math.sqrt(self.n)
        self.T = 4 * self.sigma * (math.log(self.k) + math.log(1 / self.beta))

        self.histogram = Histogram(num_features, domain_size_per_feature, domain_size)

    def dump(
        self,
    ):
        histogram = yaml.dump(self.histogram)
        print("Histogram", histogram)

    def run_cache(self, query_id, blocks, budget):
        # Runs the PMW algorithm
        for round in range(self.k):
            # Compute the true noisy output
            noise_sample = np.random.laplace(scale=self.sigma)

            def translate_laplace_noise_to_epsilon(sigma):
                pass

            budget = translate_laplace_noise_to_epsilon(self.sigma)
            true_output = self.scheduler.run_task(
                query_id, blocks, budget
            )  # Don't waste budget just yet
            predicted_output = self.histogram.run_task(query_id)
            ###############################################
            # Compute the error
            # Is the query hard
            # Update or/and output etc
            # ............
            ###############################################

    def get_execution_plan(self, query_id, blocks, budget):
        """
        For per-block-pmu all plans have this form: A(F(B1), F(B2), ... , F(Bn))
        """
        num_aggregations = len(blocks)
        plan = []
        splits = get_splits(blocks, num_aggregations)
        for split in splits:
            # print("split", split)
            for x in split:
                x = (x[0], x[-1])
                plan += [cache.F(query_id, x, budget)]
                return cache.A(plan)
        return None


def main():
    num_features = 2
    domain_size_per_feature = {"new_cases": 2, "new_deaths": 2}

    domain_size = 1
    for v in domain_size_per_feature.values():
        domain_size *= v

    histogram = Histogram(num_features, domain_size_per_feature, domain_size)

    # self.n = 100         # block size
    # self.epsilon = 0.1
    # self.delta = 0.01
    # self.beta = 0.001
    # self.M = domain_size
    # self.k = 100


if __name__ == "__main__":
    main()
