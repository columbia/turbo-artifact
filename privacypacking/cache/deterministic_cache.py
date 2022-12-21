from privacypacking.cache.cache import Cache
from privacypacking.budget.block import HyperBlock
from privacypacking.budget.utils import from_pure_epsilon_to_budget
import yaml

class DeterministicCache(Cache):
    def __init__(self, variance_reduction):
        self.key_values = {}
        self.variance_reduction = variance_reduction

    def add_entry(self, query_id, hyperblock_id, result, budget, noise):
        if query_id not in self.key_values:
            self.key_values[query_id] = {}
        self.key_values[query_id].update({hyperblock_id: (result, budget, noise)})

    def get_entry(self, query_id, hyperblock_id):
        if query_id in self.key_values:
            if hyperblock_id in self.key_values[query_id]:
                (result, budget, noise) = self.key_values[query_id][hyperblock_id]
                return result, budget, noise
        return None, None, None

    def run(self, query_id, query, demand_budget, hyperblock: HyperBlock):
        run_budget = None

        true_result, cached_budget, cached_noise = self.get_entry(
            query_id, hyperblock.id
        )
        if true_result is None:  # Not cached ever
            true_result = hyperblock.run(query)  # Run without noise
            run_budget = demand_budget
            noise = run_budget.compute_noise()
        else:  # Cached already with some budget and noise
            if (
                demand_budget.pure_epsilon <= cached_budget.pure_epsilon
            ):  # If cached budget is enough
                noise = cached_noise
            else:  # If cached budget is not enough
                if self.variance_reduction:  # If optimization is enabled
                    run_budget = from_pure_epsilon_to_budget(demand_budget.pure_epsilon-cached_budget.pure_epsilon)
                    run_noise = run_budget.compute_noise()
                    noise = (
                        cached_budget.pure_epsilon * cached_noise
                        + run_budget.pure_epsilon * run_noise
                    ) / (cached_budget + run_budget).pure_epsilon
                else:  # If optimization is not enabled
                    run_budget = demand_budget
                    noise = run_budget.compute_noise()

        result = true_result + noise

        if run_budget is not None:
            self.add_entry(query_id, hyperblock.id, true_result, demand_budget, noise)
        return result, run_budget


    def dump(self):
        res = yaml.dump(self.key_values)
        print("Results", res)
