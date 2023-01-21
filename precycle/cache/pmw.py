import torch
import numpy as np
from loguru import logger

from budget import Budget
from budget.histogram import DenseHistogram
from budget.curves import (
    BoundedOneShotSVT,
    GaussianCurve,
    LaplaceCurve,
    PureDPtoRDP,
    ZeroCurve,
)
from utils.utils import mlflow_log
from precycle.tesnor_converter import TensorConverter



# TODO: what is the minimum info that has to be stored in Redis so that I can restore the PMW?
class PMW:
    def __init__(
        self,
        blocks,
        blocks_metadata,
        nu=None,  # Scale of noise added on queries. Should be computed from alpha.
        ro=None,  # Scale of noise added on the threshold. Will be nu if left empty. Unused for Laplace SVT.
        alpha=0.005,  # Max error guarantee, expressed as fraction.
        beta=0.01,  # Failure probability for the alpha error bound
        k=None,  # Max number of queries for each OneShot SVT instance. Unused for Laplace SVT
        standard_svt=True,  # Laplace SVT by default. Gaussian RDP SVT otherwise
        output_counts=True,  # False to output fractions (like PMW), True to output raw counts (like MWEM)
    ):
        # TODO: some optimizations
        # - a friendlier constructor computes nu based on a fraction of block budget (and alpha)
        # - for unlimited kmax, nonnegative queries -> RDP SVT gives a (maybe) tighter theorem. But let's stay "simple" for now.
        # - cheap version of the previous point: dynamic k, increase over time
        # - MWEM makes bigger steps when the error is higher, we could try that too
        # - Is it better to use Laplace after hard queries, for composition? By how much?

        self.blocks_metadata = blocks_metadata

        # Generic PMW arguments
        self.n = sum(
            [
                float(blocks_metadata["blocks"][str(id)]["size"])
                for id in range(blocks[0], blocks[1] + 1)
            ]
        )
        self.M = float(blocks_metadata["domain_size"])
        self.k = k  # max_total_queries
        self.queries_ran = 0
        self.hard_queries_ran = 0
        self.histogram = DenseHistogram(self.M)
        self.id = str(blocks)[1:-1].replace(", ", "-")
        self.output_counts = output_counts

        # From my maths. It's cheap to be accurate when n is large.
        self.nu = nu if nu else self.n * alpha / np.log(2 / beta)

        # Sparse Vector parameters
        self.alpha = alpha
        self.local_svt_queries_ran = 0
        self.local_svt_max_queries = k
        self.ro = ro if ro else nu
        self.standard_svt = standard_svt

        # Always sensitivity 1/n, if we want counts we'll scale as post-processing
        self.Delta = 1 / self.n

        # The initial threshold should be noisy too if we want to use Sparse Vector
        if self.standard_svt:
            # ro=1/eps1 and nu=1/eps2
            self.noisy_threshold = self.alpha / 2 + np.random.laplace(
                0, self.Delta * self.nu
            )
        else:
            self.noisy_threshold = self.alpha / 2 + np.random.normal(
                0, self.Delta * self.ro
            )

    def worst_case_cost(self) -> Budget:
        # Worst case: we need to pay for a new sparse vector (e.g. first query, or first query after cache miss)
        # and we still do a cache miss, so we pay for a true query on top of that
        # query_cost = GaussianCurve(sigma=nu)
        query_cost = LaplaceCurve(laplace_noise=self.nu)
        if self.standard_svt:
            # We add only noise 1/nu before comparing to the threshold, so it costs 2/nu (see Salil)
            svt_cost = PureDPtoRDP(epsilon=1 / self.nu + 2 / self.nu)
        else:
            svt_cost = BoundedOneShotSVT(
                ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
            )
        return svt_cost + query_cost

    # TODO: this heuristic is a toy example that I use as a mock up.
    def predict_hit(self):
        """A heuristic that tries to predict whether we will have a miss or hit based on the past number of hits"""
        # threshold = 30
        # if self.hard_queries_ran - self.queries_ran > threshold:
        # return 1
        # for now only returns worst case (miss)
        return 0

    def estimate_run_budget(self) -> Budget:
        return ZeroCurve() if self.predict_hit() else self.worst_case_cost()

    def run(self, query, true_output):
        self.tensor_convertor = TensorConverter(self.blocks_metadata)
        assert isinstance(query, torch.Tensor)

        run_metadata = {}

        if (not self.standard_svt) and (
            self.local_svt_queries_ran >= self.local_svt_max_queries
        ):
            # The Laplace sparse vector can run forever as long as queries are easy
            self.local_svt_queries_ran = 0
            logger.warning(
                "Local sparse vector is exhausted (too many easy queries). Starting a new one..."
            )

        # Pay the initialization budget if it's the first call
        if self.local_svt_queries_ran == 0:

            if self.standard_svt:
                self.noisy_threshold = self.alpha / 2 + np.random.laplace(
                    0, self.Delta * self.nu
                )
                run_budget = PureDPtoRDP(epsilon=1 / self.nu + 2 / self.nu)
            else:
                self.noisy_threshold = self.alpha / 2 + np.random.normal(
                    0, self.Delta * self.ro
                )
                run_budget = BoundedOneShotSVT(
                    ro=self.ro, nu=self.nu, kmax=self.local_svt_max_queries
                )
        else:
            run_budget = ZeroCurve()

        # Comes for free (public histogram). Always normalized, outputs fractions
        predicted_output = self.histogram.run(query)
        if self.output_counts:
            # Executor doesn't run normalized queries
            true_output /= self.n

        self.queries_ran += 1
        self.local_svt_queries_ran += 1

        # `noisy_error` is a DP query with sensitivity self.Delta, Sparse Vector needs twice that
        true_error = abs(true_output - predicted_output)
        error_noise = (
            np.random.laplace(
                0, self.Delta * self.nu
            )  # NOTE: Factor goes to the budget instead
            if self.standard_svt
            else np.random.normal(0, 2 * self.Delta * self.nu)
        )
        noisy_error = true_error + error_noise

        if noisy_error < self.noisy_threshold:
            # Easy query, i.e. "Output bot" in SVT
            run_metadata["hard_query"] = False
            logger.info(
                f"Easy query - Predicted: {predicted_output}, true: {true_output}, true error: {true_error}, noisy error: {noisy_error}, noise std: {2 * self.Delta * self.nu}"
            )
            output = predicted_output
        else:
            # Hard query, i.e. "Output top" in SVT
            # We'll start a new sparse vector at the beginning of the next query (and pay for it)
            run_metadata["hard_query"] = True
            self.local_svt_queries_ran = 0
            self.hard_queries_ran += 1

            logger.info(
                f"Hard query - Predicted: {predicted_output}, true: {true_output}"
            )

            # NOTE: cut-off = 1 and pay as you go -> no limit on the number of hard queries
            # # Too many hard queries - breaking privacy. Don't update histogram or return query result.
            # if self.hard_queries_answered >= self.max_hard_queries:
            #     # TODO: what do you pay exactly here?
            #     logger.warning("The planner shouldn't let you do this.")
            #     return None, run_budget

            # NOTE: Salil's PMW samples fresh noise here, it makes more sense I think.
            #       We could also use yet another noise scaling parameter
            if self.standard_svt:
                noisy_output = true_output + np.random.laplace(0, self.Delta * self.nu)
                run_budget += LaplaceCurve(laplace_noise=self.nu)
            else:
                noisy_output = true_output + np.random.normal(0, self.Delta * self.nu)
                run_budget += GaussianCurve(sigma=self.nu)

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
            output = noisy_output

        self.mlflow_log_run(output, true_output)
        run_metadata["true_error_fraction"] = abs(output - true_output)

        if self.output_counts:
            output *= self.n

        return output, run_budget, run_metadata



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