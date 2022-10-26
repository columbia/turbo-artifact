import math
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from termcolor import colored

from privacypacking.budget import BasicBudget
from privacypacking.cache import cache
from privacypacking.cache.utils import get_splits


class Histogram:
    def __init__(self, num_features, domain_size_per_feature, domain_size) -> None:
        self.domain_size_per_feature = domain_size_per_feature
        self.num_features = num_features
        self.domain_size = domain_size

        normalization_factor = 1 / self.domain_size
        shape = tuple([value for value in domain_size_per_feature.values()])
        # Initializing with a uniform distribution
        self.bins = np.full(shape=shape, fill_value=normalization_factor)

    def get_bins_idx(self, query_id):

        # Gives us direct access to the bins we are interested in
        # Two types of hardcoded queries for now: count new cases, count new deaths
        # histogram arrangement: p:positive, d:deceased
        # [[p0-d0, p0-d1],
        #  [p1-d0, p1-d1]]
        # todo: generalize

        if (
            query_id == 0
        ):  # accesses : `positive=1 AND (deceased=1 OR deceased=0)`     -- Count of New Cases
            return [(1, 0), (1, 1)]
        if (
            query_id == 1
        ):  # accesses : `(positive=1 OR positive=0) AND deceased=1`     -- Count of New Deaths
            return [(0, 1), (1, 1)]

    def get_bins(self, query_id):
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


class DenseHistogram:
    def __init__(self, attribute_sizes: List[int]) -> None:
        # TODO: optimize this later, maybe we only need to store the "diff", which is sparse
        self.N = domain_size(attribute_sizes)
        self.tensor = torch.ones(size=(1, self.N))
        self.normalize()

    def normalize(self) -> None:
        F.normalize(self.tensor, p=1, out=self.tensor)

    def run_query(self, query: torch.Tensor) -> float:
        # sparse (1,N) x dense (N,1)
        return torch.smm(query, self.tensor.t()).item()


class SparseHistogram:
    def __init__(self, attribute_sizes: List[int]) -> None:
        # Flat representation of shape (1, N)
        self.tensor = build_sparse_tensor(
            bin_indices=[],
            values=[],
            attribute_sizes=attribute_sizes,
        )

    def run_query(self, query: torch.Tensor) -> float:
        # `query` has shape (1, N), we need the dot product, or matrix mult with (1,N)x(N,1)
        return torch.mm(self.tensor, query.t()).item()


def get_flat_bin_index(
    multidim_bin_index: List[int], attribute_sizes: List[int]
) -> int:
    index = 0
    size = 1
    # Row-major order like PyTorch (inner rows first)
    for dim in range(len(attribute_sizes) - 1, -1, -1):
        index += multidim_bin_index[dim] * size
        size *= attribute_sizes[dim]
    return index


# TODO: write the inverse conversion


def domain_size(attribute_sizes: List[int]) -> int:
    domain_size = 1
    for s in attribute_sizes:
        domain_size *= s
    return domain_size


def build_sparse_tensor(
    bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
):
    # One row only
    column_ids = []
    for b, v in zip(bin_indices, values):
        column_ids.append(get_flat_bin_index(b, attribute_sizes))

    return torch.sparse_coo_tensor(
        [[0] * len(column_ids), column_ids],
        values,
        size=(1, domain_size(attribute_sizes)),
    )


def build_sparse_tensor_multidim(
    bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
):
    return torch.sparse_coo_tensor(
        list(zip(*bin_indices)), values, size=attribute_sizes
    )


class PMW:
    def __init__(
        self,
        scheduler,
        blocks,
        epsilon=0.1,
        delta=1e-4,
        beta=1e-3,
        k=100,
        attribute_sizes=(2, 2),
    ):
        self.scheduler = scheduler
        self.blocks = blocks
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.attribute_sizes = attribute_sizes
        self.k = k
        self.M = domain_size(attribute_sizes)

        # Initializations as per the Hardt and Rothblum 2010 paper
        self.w = 0
        self.sigma = (
            (10 * math.log(1 / self.delta) * math.pow(math.log(self.M), 1 / 4))
            / math.sqrt(self.n)
            * self.epsilon
        )
        self.learning_rate = math.pow(math.log(self.M), 1 / 4) / math.sqrt(self.n)
        self.T = 4 * self.sigma * (math.log(self.k) + math.log(1 / self.beta))

        self.histogram = DenseHistogram(attribute_sizes)
        # We consume the whole budget once at initialization
        self.scheduler.consume_budgets(blocks, BasicBudget(self.epsilon))

    def dump(
        self,
    ):
        # histogram = yaml.dump(self.histogram)
        print("Histogram", self.histogram)

    def is_query_hard(self, error):
        return abs(error) > self.T

    def run_cache(self, query_id, blocks, _):
        # Too many rounds - breaking privacy
        if self.queries_ran >= self.k:
            exit(0)

        # Runs the PMW algorithm
        self.queries_ran += 1
        # Compute the true noisy output
        noise_sample = np.random.laplace(scale=self.sigma)

        # TODO: use efficient dot product here (blocks as sparse histograms)
        true_output = self.scheduler.run_task(
            query_id, blocks, budget=None, disable_dp=True
        )

        # TODO: get the query (SparseHistogram) itself from the task type
        predicted_output = self.histogram.run_task(query_id)

        # TODO: log error
        # print(
        #     colored(
        #         f"\tNoisy True Result of query {query_id} on blocks {blocks}: {true_output}",
        #         "magenta",
        #     )
        # )
        # print(
        #     colored(
        #         f"\tNoisy Predicted Result of query {query_id} on blocks {blocks}: {predicted_output}",
        #         "grey",
        #     )
        # )

        error = true_output - predicted_output

        if self.is_query_hard(error):  # Is the query hard
            indices_reached = self.histogram.get_bins_idx(
                query_id
            )  # get the indices that are "reached" by the query
            copy_histogram = deepcopy(self.histogram)

            # TODO(P1): write as proper vector operations

            ### UPDATING IS TAILORED ONLY FOR COUNT QUERIES NOW ##
            if error > 0:
                # r_i is 1 for reached indices and 0 for unreached -- update only for reached indices - unreached remain the same
                copy_histogram.bins[indices_reached] *= math.exp(-self.learning_rate)
            else:
                # r_i is 0 for reached indices and 1 for unreached -- update only for unreached indices - reached remain the same
                copy_histogram.bins *= math.exp(-self.learning_rate)
                copy_histogram.bins[indices_reached] = self.histogram.bins[
                    indices_reached
                ]  # Re-write original values to reached indices

            # Now we need to normalize1
            normalizing_factor = np.sum(copy_histogram.bins)  # reduces to a scalar
            copy_histogram.bins *= 1 / normalizing_factor

            self.histogram.normalize()

            # Too many hard queries - breaking privacy
            if self.w > self.n * math.pow(math.log(self.M), 1 / 2):
                exit(0)

            self.w += 1
            return true_output

        else:
            return predicted_output


# This is for one instance of PMW
class PMW_demo:
    def __init__(self, scheduler, blocks):

        self.scheduler = scheduler
        self.blocks = blocks
        self.n = (blocks[1] - blocks[0] + 1) * scheduler.block_size  # data size

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
            print(
                colored(
                    "\tHard Query - oh no!",
                    "red",
                )
            )
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
        true_output = (
            self.scheduler.run_task(query_id, blocks, budget=None, disable_dp=True)
            + noise_sample
        )  # but don't waste budget just yet!!
        predicted_output = self.histogram.run_task(query_id)

        print(
            colored(
                f"\tNoisy True Result of query {query_id} on blocks {blocks}: {true_output}",
                "magenta",
            )
        )
        print(
            colored(
                f"\tNoisy Predicted Result of query {query_id} on blocks {blocks}: {predicted_output}",
                "grey",
            )
        )

        # Compute the error
        error = true_output - predicted_output
        # print(error)
        if self.is_query_hard(error):  # Is the query hard
            indices_reached = self.histogram.get_bins_idx(
                query_id
            )  # get the indices that are "reached" by the query
            copy_histogram = deepcopy(self.histogram)

            ### UPDATING IS TAILORED ONLY FOR COUNT QUERIES NOW ##
            if error > 0:
                # r_i is 1 for reached indices and 0 for unreached -- update only for reached indices - unreached remain the same
                copy_histogram.bins[indices_reached] *= math.exp(-self.learning_rate)
            else:
                # r_i is 0 for reached indices and 1 for unreached -- update only for unreached indices - reached remain the same
                copy_histogram.bins *= math.exp(-self.learning_rate)
                copy_histogram.bins[indices_reached] = self.histogram.bins[
                    indices_reached
                ]  # Re-write original values to reached indices

            # Now we need to normalize1
            normalizing_factor = np.sum(copy_histogram.bins)  # reduces to a scalar
            copy_histogram.bins *= 1 / normalizing_factor

            self.histogram = copy_histogram

            # Too many hard queries - breaking privacy
            if self.w > self.n * math.pow(math.log(self.M), 1 / 2):
                exit(0)

            self.w += 1
            return true_output

        else:
            return predicted_output


def debug1():

    # Testing ...
    num_features = 2
    domain_size_per_feature = {"new_cases": 2, "new_deaths": 2}

    domain_size = 1
    for v in domain_size_per_feature.values():
        domain_size *= v

    histogram = Histogram(num_features, domain_size_per_feature, domain_size)

    histogram.update_bins([(1, 0), (1, 1)], [1, 2])
    histogram.update_bins([(0, 1), (1, 1)], [3, 4])
    bins = histogram.get_bins(0)
    bins = histogram.get_bins(1)
    histogram.run_task(0)

    # self.n = 100         # block size
    # self.epsilon = 0.1
    # self.delta = 0.01
    # self.beta = 0.001
    # self.M = domain_size
    # self.k = 100


def debug2():

    attribute_sizes = [2, 2, 10]

    print(
        get_flat_bin_index(
            multidim_bin_index=(0, 0, 0), attribute_sizes=attribute_sizes
        )
    )

    print(
        get_flat_bin_index(
            multidim_bin_index=(0, 1, 2), attribute_sizes=attribute_sizes
        )
    )

    h = DenseHistogram(attribute_sizes)
    print(h.tensor)
    # q = build_sparse_tensor({(0, 0, 0): 1.0, (0, 1, 5): 1.0})

    block = build_sparse_tensor(
        bin_indices=[[0, 0, 1], [0, 1, 5], [0, 0, 0]],
        values=[1.0, 4.0, 3.0],
        attribute_sizes=attribute_sizes,
    )

    q = build_sparse_tensor(
        bin_indices=[[0, 0, 0], [0, 1, 5]],
        values=[1.0, 1.0],
        attribute_sizes=attribute_sizes,
    )
    print(q)

    print(torch.sparse.mm(block, q.t()).item())

    print(
        build_sparse_tensor(
            bin_indices=[],
            values=[],
            attribute_sizes=attribute_sizes,
        )
    )

    print(h.run_query(q))

    v = torch.sparse_coo_tensor(
        indices=[[0, 0, 0], [0, 1, 2]], values=[1.0, 1.0, 1.0], size=(1, 40)
    )
    q = torch.sparse_coo_tensor(
        indices=[[0, 0], [1, 2]], values=[3.0, 4.0], size=(1, 40)
    )
    print(torch.sparse.mm(v, q.t()).item())

    # print(h)
    # print(q)
    # print(h.run_query(q))


if __name__ == "__main__":
    debug2()
