from privacypacking.cache.utils import get_splits
from termcolor import colored
from privacypacking.cache import cache
import math
import numpy as np
import pandas as pd
from privacypacking.budget import BasicBudget
from copy import deepcopy


class Histogram:
    def __init__(self, num_features, domain_size_per_feature, domain_size) -> None:
        self.domain_size_per_feature = domain_size_per_feature
        self.num_features = num_features
        self.domain_size = domain_size

        normalization_factor = 1 / self.domain_size            
        shape = tuple([value for value in domain_size_per_feature.values()])
        # Initializing with a uniform distribution
        self.bins = np.full(shape=shape, fill_value=normalization_factor)

    def get_bins_idx(
        self, query_id
    ):

      # Gives us direct access to the bins we are interested in
        # Two types of hardcoded queries for now: count new cases, count new deaths
        # histogram arrangement: p:positive, d:deceased
        # [[p0-d0, p0-d1],
        #  [p1-d0, p1-d1]]
        # todo: generalize

        if query_id == 0:  # accesses : `positive=1 AND (deceased=1 OR deceased=0)`     -- Count of New Cases
            return [(1,0), (1,1)]
        if query_id == 1:  # accesses : `(positive=1 OR positive=0) AND deceased=1`     -- Count of New Deaths
            return [(0,1), (1,1)]

    def get_bins(
        self, query_id
    ):
        return [self.bins[idx] for idx in self.get_bins_idx(query_id)]


    def update_bins(self, indices, values):
        for idx, value in zip(indices, values):
            self.bins[idx] = value

    def run_task(self, query_id):
        return np.sum(self.get_bins(query_id))

    def dump(
        self,
    ):
        return self.bins

# This is for one instance of PMW
class PMW:
    def __init__(self, scheduler, blocks):

        self.scheduler = scheduler
        self.blocks = blocks
        self.n = (blocks[1]-blocks[0]+1) * scheduler.block_size  # data size

        #############################
        # todo: for now all this is hardcoded
        num_features = 2
        domain_size = 4
        domain_size_per_feature = {"positive": 2, "deceased": 2}
        self.epsilon = 0.1
        self.delta = 0.01
        self.beta = 0.001
        self.M = domain_size
        self.k = 100
        #############################

        self.queries_ran = 0

        # Initializations as per the Hardt and Rothblum 2010 paper
        self.w = 0
        self.sigma = (
            (10 * math.log(1 / self.delta) * math.pow(math.log(self.M), 1 / 4))
            / math.sqrt(self.n)
            * self.epsilon
        )
        self.learning_rate = math.pow(math.log(self.M), 1 / 4) / math.sqrt(self.n)
        self.T = 4 * self.sigma * (math.log(self.k) + math.log(1 / self.beta))

        self.histogram = Histogram(num_features, domain_size_per_feature, domain_size)
        # We consume the whole budget once at initialization
        self.scheduler.consume_budgets(blocks, BasicBudget(self.epsilon))

    def dump(
        self,
    ):
        # histogram = yaml.dump(self.histogram)
        print("Histogram", self.histogram)

    def is_query_hard(self, error):
        if abs(error) > self.T:
            return True
        return False

    def run_cache(self, query_id, blocks, _):
        ### NOTE:  I'm not using the budget argument because I no longer care about the user defined epsilon

        # Too many rounds - breaking privacy
        if self.queries_ran >= self.k:
            exit(0)

        # Runs the PMW algorithm
        self.queries_ran += 1
        # Compute the true noisy output
        noise_sample = np.random.laplace(scale=self.sigma)

        # adding noise directly here
        true_output = self.scheduler.run_task(
            query_id, blocks, budget=None, disable_dp=True
        )  + noise_sample # but don't waste budget just yet!!
        predicted_output = self.histogram.run_task(query_id)

        # Compute the error
        error = true_output - predicted_output
        if self.is_query_hard(error):   # Is the query hard            
            indices_reached = self.histogram.get_bins_idx(query_id)  # get the indices that are "reached" by the query                    
            copy_histogram = deepcopy(self.histogram)


            ### UPDATING IS TAILORED ONLY FOR COUNT QUERIES NOW ##
            if error > 0:
                # r_i is 1 for reached indices and 0 for unreached -- update only for reached indices - unreached remain the same
                copy_histogram.bins[indices_reached] *= math.exp(-self.learning_rate)
            else:
                # r_i is 0 for reached indices and 1 for unreached -- update only for unreached indices - reached remain the same
                copy_histogram.bins *= math.exp(-self.learning_rate)
                copy_histogram.bins[indices_reached] = self.histogram.bins[indices_reached]     # Re-write original values to reached indices

            # Now we need to normalize1
            normalizing_factor = np.sum(copy_histogram.bins)      # reduces to a scalar
            copy_histogram.bins *= 1/normalizing_factor

            self.histogram = copy_histogram


            # Too many hard queries - breaking privacy
            if self.w > self.n * math.pow(math.log(self.M), 1 / 2):
                exit(0)
            
            self.w += 1
            return true_output

        else:
            return predicted_output


def main():

    # Testing ... 
    num_features = 2
    domain_size_per_feature = {"new_cases": 2, "new_deaths": 2}

    domain_size = 1
    for v in domain_size_per_feature.values():
        domain_size *= v

    histogram = Histogram(num_features, domain_size_per_feature, domain_size)

    histogram.update_bins([(1,0), (1,1)], [1,2])
    histogram.update_bins([(0,1), (1,1)], [3,4])
    bins = histogram.get_bins(0)
    bins = histogram.get_bins(1)
    histogram.run_task(0)



    # self.n = 100         # block size
    # self.epsilon = 0.1
    # self.delta = 0.01
    # self.beta = 0.001
    # self.M = domain_size
    # self.k = 100


if __name__ == "__main__":
    main()


