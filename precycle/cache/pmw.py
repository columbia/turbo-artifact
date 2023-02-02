import numpy as np
import torch
from loguru import logger

from precycle.budget import Budget
from precycle.budget.curves import (BoundedOneShotSVT, GaussianCurve,
                                    LaplaceCurve, PureDPtoRDP, ZeroCurve)
from precycle.budget.histogram import DenseHistogram
from precycle.utils.utils import get_blocks_size, mlflow_log


# TODO: what is the minimum info that has to be stored in Redis so that I can restore the PMW?
class PMW:
    def __init__(
        self,
        blocks,
        alpha,  # Max error guarantee, expressed as fraction.
        beta,  # Failure probability for the alpha error bound
        old_pmw=None,  # PMW to initialize from
        nu=None,  # Scale of noise added on queries. Should be computed from alpha.
        ro=None,  # Scale of noise added on the threshold. Will be nu if left empty. Unused for Laplace SVT.
        k=None,  # Max number of queries for each OneShot SVT instance. Unused for Laplace SVT
        standard_svt=True,  # Laplace SVT by default. Gaussian RDP SVT otherwise
        output_counts=False,  # False to output fractions (like PMW), True to output raw counts (like MWEM)
        heuristic=None,
        heuristic_value=None,
        blocks_metadata=None,
    ):
        # TODO: some optimizations
        # - a friendlier constructor computes nu based on a fraction of block budget (and alpha)
        # - for unlimited kmax, nonnegative queries -> RDP SVT gives a (maybe) tighter theorem. But let's stay "simple" for now.
        # - cheap version of the previous point: dynamic k, increase over time
        # - MWEM makes bigger steps when the error is higher, we could try that too
        # - Is it better to use Laplace after hard queries, for composition? By how much?

        # PMW will always be initialized by alpha, beta
        assert alpha is not None and beta is not None

        self.blocks_metadata = blocks_metadata

        # Generic PMW arguments
        # Assuming all blocks have the same size for now
        self.n = get_blocks_size(blocks, blocks_metadata)
        self.M = blocks_metadata["domain_size"]
        self.k = k  # max_total_queries
        self.queries_ran = 0
        self.hard_queries_ran = 0
        self.histogram = DenseHistogram(self.M) if not old_pmw else old_pmw.histogram
        self.id = str(blocks)[1:-1].replace(", ", "-")
        self.output_counts = output_counts

        # Heuristics
        self.heuristic = heuristic
        self.heuristic_value = heuristic_value
        self.past_queries_difficulty = [0] * self.heuristic_value
        self.pmw_updates_count = 0
        self.visits_count_histogram = torch.zeros(size=(1, self.M), dtype=torch.float64)

        # From my maths. It's cheap to be accurate when n is large.
        self.nu = nu if nu else self.n * alpha / np.log(2 / beta)

        # Sparse Vector parameters
        self.alpha = alpha
        self.beta = beta
        self.local_svt_queries_ran = 0
        self.local_svt_max_queries = k
        self.ro = ro if ro else nu
        self.standard_svt = standard_svt

        # Always sensitivity 1/n, if we want counts we'll scale as post-processing
        self.Delta = 1 / self.n

        laplace_scale = self.nu * self.Delta
        # noise std of noise added on queries. using it to check if an external update is eligible
        self.noise_std = laplace_scale * np.sqrt(2)

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

    # Heuristic 1
    def predict_hit_avg_bin_visits_heuristic(self, query):
        """A heuristic that tries to predict whether we will have a miss or hit based on the past number of hits"""

        avg_bin_visits = 0
        for i in query.indices()[1]:
            avg_bin_visits += self.visits_count_histogram[0, i]
        avg_bin_visits /= len(query.indices()[1])
        # print("avg_bin_visits", avg_bin_visits)
        if avg_bin_visits > self.heuristic_value:
            return True
        return False

    # Heuristic 2
    def predict_hit_total_updates_heuristic(self, query):
        # print(self.pmw_updates_count, self.heuristic_value)
        return self.pmw_updates_count > self.heuristic_value

    # Heuristic 3
    def predict_hit_hard_queries_heuristic(self, query):
        last_n_queries = self.past_queries_difficulty[-self.heuristic_value :]
        # print(last_n_queries)
        easy_queries = 1 - sum(last_n_queries) / self.heuristic_value
        print(easy_queries)
        if easy_queries >= 0.5:
            return True
        else:
            # Give 20% chance for the PMW to run, otherwise we will never have queries
            return np.random.choice([True, False], 1, p=[0.4, 0.6])[0]

    def estimate_run_budget(self, query) -> Budget:
        if self.heuristic == "n_past_queries":
            hit = self.predict_hit_hard_queries_heuristic(query)
        elif self.heuristic == "avg_bin_visits":
            hit = self.predict_hit_avg_bin_visits_heuristic(query)
        elif self.heuristic == "total_updates_counts":
            hit = self.predict_hit_total_updates_heuristic(query)
        return ZeroCurve() if hit else self.worst_case_cost()

    def run(self, query, true_output):
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
            self.past_queries_difficulty.append(0)
            logger.info(
                f"Easy query - Predicted: {predicted_output}, true: {true_output}, true error: {true_error}, noisy error: {noisy_error}, noise std: {2 * self.Delta * self.nu}"
            )
            output = predicted_output
        else:
            # Hard query, i.e. "Output top" in SVT
            # We'll start a new sparse vector at the beginning of the next query (and pay for it)
            run_metadata["hard_query"] = True
            self.past_queries_difficulty.append(1)
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
                self.visits_count_histogram[0, i] += 1

            self.histogram.normalize()
            self.pmw_updates_count += 1
            output = noisy_output

        self.mlflow_log_run(output, true_output)
        run_metadata["true_error_fraction"] = abs(output - true_output)

        if self.output_counts:
            output *= self.n

        return output, run_budget, run_metadata

    def external_update(self, query, noisy_result):
        # Important: External updates probably break proofs
        predicted_output = self.histogram.run(query)

        # Multiplicative weights update for the relevant bins
        values = query.values()
        if noisy_result > predicted_output:
            # We need to make the estimated count higher to be closer to reality
            updates = torch.exp(values * self.alpha / 8)
        else:
            updates = torch.exp(-values * self.alpha / 8)
        for i, u in zip(query.indices()[1], updates):
            self.histogram.tensor[0, i] *= u
            self.visits_count_histogram[0, i] += 1
        self.histogram.normalize()
        self.pmw_updates_count += 1

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
