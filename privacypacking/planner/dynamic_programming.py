from privacypacking.planner.planner import Planner
from privacypacking.cache.cache import A, R
import numpy as np
import math

class DPlanner(Planner):
    def __init__(self, cache, blocks):
        super().__init__(cache)
        self.blocks = blocks
    
    def create_table(self, query_id, budget):
        n = len(self.blocks)
        self.cost_table = np.full([n,n], np.inf)
        self.path_table = []
        for i in range(n):
            self.path_table.append([None for _ in range(n-i)])

        for i in range(n):    
            for j in range(n-i):
                blocks = (j,j+i)

                # Exclude cells that do not satisfy the structure constraint
                Cij_plan = R(query_id, blocks, budget) # Cost of [j,j+i] with no cuts
                Cij = get_cost(Cij_plan, blocks, self.blocks, budget)
                print("no cuts plan", Cij_plan, "cost", Cij)
                costs = [Cij]
                paths = [Cij_plan]
                for k in range(i): 
                    b1 = (j, j+k)
                    b2 = (j+k+1, j+i)
                    plan = A([R(query_id, b1, budget), R(query_id, b2, budget)])
                    # print("cut", k, plan)
                    # print("idx:", (k,j), (i-k-1, j+k+1))
                    cost = self.cost_table[k][j] + self.cost_table[i-k-1][j+k+1]
                    path = ((k,j), (i-k-1,j+k+1))
                    costs.append(cost)
                    paths.append(path)

                # print("custcosts", costs)
                idx = np.argmin(np.array(costs))
                Fij = costs[idx]
                Pij = paths[idx]

                self.cost_table[i][j] = Fij
                self.path_table[i][j] = Pij

    def get_plan_from_path(self, request):
        i = request[0]
        j = request[1]
        plan = self.path_table[i][j]
        if isinstance(plan, R):
            return [plan]
        else:
            plan_a = self.get_plan_from_path(plan[0])
            plan_b = self.get_plan_from_path(plan[1])
            return plan_a + plan_b


def satisfies_constraint(blocks):
    bf = 2  # Branching factor
    size = blocks[1]-blocks[0]+1
    if not math.log(size, bf).is_integer():
        return False
    if size > 1 and not (blocks[1] % bf):
        return False
    return True

def mapping_blocks_to_cell(blocks):
    # (i,j) --> (j, j+i) mapping from a cell to the blocks it represents
    j = blocks[0]
    i = blocks[1]-j        
    return (i,j)

def get_cost(plan, requested_blocks, blocks, budget):
    if not satisfies_constraint(requested_blocks):
        return np.inf
    return 1


def test():
    blocks = [0,1,2,3,4,5]
    dplanner = DPlanner(None, blocks)
    dplanner.create_table(0, 0)
    
    n = len(blocks)
    for i in range(n):
        for j in range(n-i):
            print(dplanner.cost_table[i][j], end =" ")
        print("\n")

    for i in range(n):
        for j in range(n-i):
            print(dplanner.path_table[i][j], end =" ")
        print("\n")


    request = (0,1)
    idx = mapping_blocks_to_cell(request)
    plan = A(dplanner.get_plan_from_path(idx))
    print(f"Get plan for blocks {request}: {plan}")

    request = (0,4)
    idx = mapping_blocks_to_cell(request)
    plan = A(dplanner.get_plan_from_path(idx))
    print(f"Get plan for blocks {request}: {plan}")


if __name__ == "__main__":
    test()
