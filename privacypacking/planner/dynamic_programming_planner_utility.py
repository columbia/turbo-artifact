from privacypacking.planner.planner import Planner
from privacypacking.cache.cache import A, R
from privacypacking.budget import BasicBudget
import numpy as np
import math


class DynamicProgrammingPlannerUtility(Planner):
    def __init__(self, cache, blocks, branching_factor, utility):
        super().__init__(cache)
        self.blocks = blocks
        self.cost_table = None
        self.path_table = None
        self.branching_factor = branching_factor
        self.delta = 0.00001
        self.utility = utility
        self.emax = 0.5

    def get_execution_plan(self, query_id, blocks, _):
        plan = None
        n = len(blocks)
        blocks = (blocks[0], blocks[-1])

        def min_epsilon(k, delta, u):
            return (math.sqrt(8*k)*math.log(2/delta)) / u

        ek_pairs = {}
        for k in range(1, n+1):
            ek_pairs[k] = min_epsilon(k, self.delta, self.utility)
        # print("EK pairs", ek_pairs)

        for k,e in ek_pairs.items():
            # print("e", e, "k", k)
            if e > self.emax:
                return plan

            budget = BasicBudget(e)
            self.create_table(query_id, blocks, budget)
            idx = mapping_blocks_to_cell(blocks)
            bestCost = self.get_cost_from_costs(idx)
            # print("bestCost:", bestCost, "K:", k)
            if bestCost <= k:
                plan = self.get_plan_from_path(idx)
                if plan is not None:
                    plan = A(plan, budget)
                    print(f"Got plan for blocks {blocks}: {plan}, {plan.budget}")
                    return plan
        return plan

    def create_table(self, query_id, block_request, budget):
        n = len(self.blocks)
        self.cost_table = np.full([n, n], math.inf)
        self.path_table = []
        for i in range(n):
            self.path_table.append([None for _ in range(n - i)])

        # for i in range(n):
        #     for j in range(n - i):
        start = block_request[0]
        end = block_request[1]+1
        for i in range(end-start):
            for j in range(start, end - i):
                blocks = (j, j + i)
                Cij_plan = R(query_id, blocks, budget)  # Cost of [j,j+i] with no cuts
                Cij = self.cache.get_cost(Cij_plan, self.blocks, True, self.branching_factor)
                costs = [Cij]
                paths = [Cij_plan]
                for k in range(i):
                    cost = self.cost_table[k][j] + self.cost_table[i - k - 1][j + k + 1]
                    path = ((k, j), (i - k - 1, j + k + 1))
                    costs.append(cost)
                    paths.append(path)

                idx = np.argmin(np.array(costs))
                if not math.isinf(costs[idx]):
                    self.cost_table[i][j] = costs[idx]
                    self.path_table[i][j] = paths[idx]


    def get_cost_from_costs(self, request):
        i = request[0]
        j = request[1]
        return self.cost_table[i][j]

    def get_plan_from_path(self, request):
        i = request[0]
        j = request[1]
        plan = self.path_table[i][j]
        if plan is None:
            return None
        elif isinstance(plan, R):
            return [plan]
        else:
            plan_a = self.get_plan_from_path(plan[0])
            plan_b = self.get_plan_from_path(plan[1])
            return plan_a + plan_b


def mapping_blocks_to_cell(blocks):
    # (i,j) --> (j, j+i) mapping from a cell to the blocks it represents
    j = blocks[0]
    i = blocks[1] - j
    return (i, j)


def get_cost(plan, requested_blocks, blocks, budget):
    return 1


def test():
    def run(request):
        blocks = [0, 1, 2, 3, 4, 5]
        dplanner = DynamicProgrammingPlannerUtility(None, blocks)       
        dplanner.create_table(0, request, 0)
        n = len(blocks)
        for i in range(n):
            for j in range(n - i):
                print(dplanner.cost_table[i][j], end=" ")
            print("\n")

        for i in range(n):
            for j in range(n - i):
                print(dplanner.path_table[i][j], end=" ")
            print("\n")
        idx = mapping_blocks_to_cell(request)
        plan = A(dplanner.get_plan_from_path(idx))
        print(f"Get plan for blocks {request}: {plan}")

    request = (0, 1)
    run(request)
    request = (0, 4)
    run(request)
    request = (2, 4)
    run(request)    

if __name__ == "__main__":
    test()
