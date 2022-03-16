import argparse

from privacypacking.budget import SubsampledGaussianCurve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", dest="sigma", type=float)
    args = parser.parse_args()

    return SubsampledGaussianCurve.from_training_parameters(
        dataset_size=50_000, batch_size=100, epochs=100, sigma=args.sigma
    ).epsilons


if __name__ == "__main__":
    demands = main()
    for demand in demands:
        print("- ", demand)