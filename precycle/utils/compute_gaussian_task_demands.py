import math
import argparse


def compute_gaussian_demands(epsilon, delta):
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

    def gaussian_dp2sigma(sensitivity=1):
        return (sensitivity / epsilon) * math.sqrt(2 * math.log(1.25 / delta))

    def compute_rdp_epsilons_gaussian(sigma):
        return [alpha / (2 * (sigma**2)) for alpha in alphas]

    return compute_rdp_epsilons_gaussian(gaussian_dp2sigma())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", dest="epsilon", type=float)
    parser.add_argument("--delta", dest="delta", type=float)
    args = parser.parse_args()

    return compute_gaussian_demands(args.epsilon, args.delta)


if __name__ == "__main__":
    demands = main()
    for demand in demands:
        print("- ", demand)
