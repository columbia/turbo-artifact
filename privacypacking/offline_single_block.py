"""
Comparing DPF vs Packing
Offline, single block setting
"""

from privacypacking.budget import Budget
from privacypacking.curves import GaussianBudget


def pack(demand_vectors, capacity):
    pass


def main():
    block = Budget.from_epsilon_delta(epsilon=10, delta=0.001)

    job = GaussianBudget(sigma=1.0)

    print(block)

    print(block - job)

    print((block - job).is_positive())


if __name__ == "__main__":
    main()
