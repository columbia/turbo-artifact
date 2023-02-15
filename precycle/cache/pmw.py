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
from precycle.budget.histogram import DenseHistogram, flat_indices
from precycle.utils.utils import get_blocks_size, mlflow_log

"""
Trimmed-down implementation of PMW, following Salil's pseudocode
"""


class PMW:
    def __init__(
        self,
        alpha,  # Max error guarantee, expressed as fraction.
        epsilon,  # Not the global budget - internal Laplace will be Lap(1/(epsilon*n))
        n,  # Number of samples
        domain_size,  # From blocks_metadata
        old_pmw=None,  # PMW to initialize from
        heuristic="bin_visits:100-1",
        id="",  # Name to log results
        max_external_updates=0,  # Deactivated by default
        external_updates_gamma=2,
        warmup_lambda=100,
    ):

        # Core PMW parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.domain_size = domain_size
        self.histogram = (
            DenseHistogram(domain_size) if not old_pmw else old_pmw.histogram
        )
        self.b = 1 / (self.n * self.epsilon)

        # Logging
        self.queries_ran = 0
        self.hard_queries_ran = 0
        self.id = id

        # Updates and warmup
        self.external_updates_count = 0
        self.max_external_updates = max_external_updates
        self.external_updates_gamma = external_updates_gamma
        self.warmup_lambda = warmup_lambda

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
        self.visits_count_histogram = torch.zeros(
            size=(1, self.domain_size), dtype=torch.float64
        )
        self.heuristic_threshold_histogram = (
            torch.ones(size=(1, self.domain_size), dtype=torch.float64)
            * self.heuristic_value
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
            output = true_output + np.random.laplace(loc=0, scale=self.b)
            run_budget += LaplaceCurve(laplace_noise=self.nu)

            # Increase weights iff predicted_output is too small
            lr = self.alpha / 8
            if output < predicted_output:
                lr *= -1

            # Multiplicative weights update for the relevant bins
            for i in flat_indices(query):
                self.histogram.tensor[i] *= torch.exp(query[i] * lr)
            self.histogram.normalize()

            # We'll start a new sparse vector at the beginning of the next query (and pay for it)
            run_metadata["hard_query"] = True
            self.local_svt_queries_ran = 0
            self.hard_queries_ran += 1
            self.pmw_updates_count += 1

        run_metadata["true_error_fraction"] = abs(output - true_output)
        return output, run_budget, run_metadata

    def external_update(self, query, noisy_result):
        if self.external_updates_count >= self.max_external_updates:
            logger.debug("Too many external updates.")
            return

        predicted_output = self.histogram.run(query)
        error = abs(predicted_output - noisy_result)
        if (
            error
            < (self.external_updates_gamma / (self.n * self.epsilon)) + self.alpha / 2
        ):
            logger.debug(
                "Skipping the external update because the histogram is accurate"
            )
            return

        # Regular multiplicative weights update
        lr = self.alpha / 8
        if noisy_result < predicted_output:
            lr *= -1
        for i in flat_indices(query):
            self.histogram.tensor[i] *= torch.exp(query[i] * lr)
        self.histogram.normalize()

        self.external_updates_count += 1

    # Heuristic 1
    def predict_hit_bin_visits_heuristic(self, query):
        for i in flat_indices(query):
            if self.visits_count_histogram[i] < self.heuristic_threshold_histogram[i]:
                return False
        return True

    # Heuristic 2
    def predict_hit_total_updates_heuristic(self, query):
        # print(self.pmw_updates_count, self.heuristic_value)
        return self.pmw_updates_count > self.heuristic_value

    # Call the heuristic
    def is_query_hard(self, query) -> bool:
        if self.heuristic_method == "bin_visits":
            return not self.predict_hit_bin_visits_heuristic(query)
        elif self.heuristic_method == "total_updates_counts":
            return not self.predict_hit_total_updates_heuristic(query)

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
