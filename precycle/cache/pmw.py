import numpy as np
import torch
from loguru import logger

from precycle.budget import Budget
from precycle.budget.curves import (
    BoundedOneShotSVT,
    GaussianCurve,
    LaplaceCurve,
    PureDPtoRDP,
    ZeroCurve,
)
from precycle.budget.histogram import DenseHistogram
from precycle.utils.utils import get_blocks_size, mlflow_log

"""
Trimmed-down implementation of PMW, following Salil's pseudocode
"""


class PMW:
    def __init__(
        self,
        blocks,
        alpha,  # Max error guarantee, expressed as fraction.
        epsilon,  # Not the global budget - internal Laplace will be Lap(1/(epsilon*n))
        blocks_metadata=None,  # Useful if all blocks have the same size
        old_pmw=None,  # PMW to initialize from
        heuristic=None,  # Defaults to threshold?
    ):

        # Core PMW parameters
        self.n = get_blocks_size(blocks, blocks_metadata)
        self.M = blocks_metadata["domain_size"]
        self.histogram = DenseHistogram(self.M) if not old_pmw else old_pmw.histogram
        self.epsilon = epsilon
        self.b = 1 / (self.n * self.epsilon)

        # Logging
        self.queries_ran = 0
        self.hard_queries_ran = 0
        self.id = str(blocks)[1:-1].replace(", ", "-")

        # Heuristics
        self.heuristic_method, self.heuristic_value = heuristic.split(":")
        if self.heuristic_method == "bin_visits":
            (
                self.heuristic_value,
                self.heuristic_value_increase,
            ) = self.heuristic_value.split("-")
            self.heuristic_value_increase = int(self.heuristic_value_increase)
        self.heuristic_value = int(self.heuristic_value)
        self.pmw_updates_count = 0
        self.visits_count_histogram = torch.zeros(size=(1, self.M), dtype=torch.float64)
        self.heuristic_threshold_histogram = (
            torch.ones(size=(1, self.M), dtype=torch.float64) * self.heuristic_value
        )

        # Initialize the first sparse vector
        self.noisy_threshold = self.alpha / 2 + np.random.laplace(loc=0, scale=self.b)
        self.local_svt_queries_ran = (
            0  # We will pay the initialization fee right before we use it
        )

    def run(self, query, true_output):
        assert isinstance(query, torch.Tensor)

        run_metadata = {}
        run_budget = ZeroCurve()

        # Pay the initialization budget if it's the first call
        if self.local_svt_queries_ran == 0:
            self.noisy_threshold = self.alpha / 2 + np.random.laplace(
                0, self.Delta * self.nu
            )
            run_budget += PureDPtoRDP(epsilon=1 / self.nu + 2 / self.nu)

        # Check the public histogram for free. Always normalized, outputs fractions
        predicted_output = self.histogram.run(query)

        # Add the sparse vector noise
        true_error = abs(true_output - predicted_output)
        error_noise = np.random.laplace(loc=0, scale=self.b)
        noisy_error = true_error + error_noise
        self.queries_ran += 1
        self.local_svt_queries_ran += 1

        # Do the sparse vector check
        if noisy_error < self.noisy_threshold:
            # Easy query, just output the histogram prediction
            output = predicted_output

            run_metadata["hard_query"] = False
        else:
            # Hard query, run a fresh Laplace estimate
            noisy_output = true_output + np.random.laplace(loc=0, scale=self.b)
            output = noisy_output
            run_budget += LaplaceCurve(laplace_noise=self.nu)

            # Multiplicative weights update for the relevant bins
            if noisy_output > predicted_output:
                # We need to make the estimated count higher to be closer to reality
                lr = self.alpha / 8
            else:
                lr = -self.alpha / 8

            # Only update the bins that were queried

            print(f"query indices: {query.indices()}")

            1 / 0

            for i in query.indices():
                self.histogram.tensor[0, i] *= torch.exp(query[i] * lr)

            # We'll start a new sparse vector at the beginning of the next query (and pay for it)
            run_metadata["hard_query"] = True
            self.local_svt_queries_ran = 0
            self.hard_queries_ran += 1

    def mlflow_log_run(self, output, true_output):
        mlflow_log(f"{self.id}/queries_ran", self.queries_ran, self.queries_ran)
        mlflow_log(
            f"{self.id}/hard_queries_ran", self.hard_queries_ran, self.queries_ran
        )
        mlflow_log(
            f"{self.id}/true_error_fraction",
            abs(output - true_output),
            self.queries_ran,
        )
        mlflow_log(
            f"{self.id}/true_error_count",
            self.n * abs(output - true_output),
            self.queries_ran,
        )
