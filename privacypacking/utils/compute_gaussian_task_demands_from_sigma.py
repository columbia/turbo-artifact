import math
import argparse


def compute_gaussian_demands(sigma):
    alphas = [
        1.5,
        1.75,
        2,
        2.5,
        3,
        4,
        5,
        6,
        8,
        16,
        32,
        64,
    ]

    def compute_rdp_epsilons_gaussian(sigma):
        return [alpha / (2 * (sigma ** 2)) for alpha in alphas]

    return compute_rdp_epsilons_gaussian(sigma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", dest="sigma", type=float)
    args = parser.parse_args()

    return compute_gaussian_demands(args.sigma)


if __name__ == "__main__":
    demands = main()
    for demand in demands:
        print("- ", demand)
