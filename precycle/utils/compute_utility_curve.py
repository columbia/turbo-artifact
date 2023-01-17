import math


def compute_utility_curve(utility, utility_beta, n):
    # TODO: replace by generic std?
    def min_epsilon(k, b, u):
        return (math.sqrt(8 * k) * math.log(2 / b)) / u

    return min_epsilon(n, utility_beta, utility)
