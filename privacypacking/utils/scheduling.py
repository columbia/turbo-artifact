from typing import Dict, List

from privacypacking.budget import Block, Task


def dominant_shares(task: Task, blocks: Dict[int, Block]) -> List[float]:
    demand_fractions = []
    for block_id, demand_budget in task.budget_per_block.items():
        block = blocks[block_id]
        block_initial_budget = block.initial_budget

        # Compute the demand share for each alpha of the block
        for alpha in block_initial_budget.alphas:

            # Drop RDP orders that are already negative
            if block_initial_budget.epsilon(alpha) > 0:
                demand_fractions.append(
                    demand_budget.epsilon(alpha) / block_initial_budget.epsilon(alpha)
                )

    # Order by highest demand fraction first
    demand_fractions.sort(reverse=True)
    return demand_fractions
