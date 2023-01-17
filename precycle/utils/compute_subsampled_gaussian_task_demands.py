import argparse

from privacypacking.budget import SubsampledGaussianCurve


def compute_subsampled_gaussian_task_demands(sigma):
    return SubsampledGaussianCurve.from_training_parameters(
        dataset_size=50_000, batch_size=100, epochs=100, sigma=sigma
    ).epsilons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", dest="sigma", type=float)
    args = parser.parse_args()

    return compute_subsampled_gaussian_task_demands(args.sigma)


if __name__ == "__main__":
    demands = main()
    for demand in demands:
        print("- ", demand)
