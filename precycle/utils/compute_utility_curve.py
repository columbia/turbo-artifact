import math


# def compute_utility_curve(utility, utility_beta, n, k):
#     def min_epsilon(u, b, n, k):
#         return (math.sqrt(8 * k) * math.log(2 / b)) / (n * u)
#     return min_epsilon(utility, utility_beta, n, k)


def deterministic_compute_utility_curve(a, b, n, n_i, k):

    """
    a: error-rate
    b: proba that error-rate will be respected
    n: total size of data requested by the user query
    k: number of computations/subqueries (aggregations: k-1)
    """

    if k >= math.log(2 / b):
        epsilon = math.sqrt(k * 8 * math.log(2 / b)) / (n * a)
    else:
        epsilon = (math.log(2 / b) * math.sqrt(8)) / (n * a)

    sensitivity = 1 / n_i
    laplace_scale = sensitivity / epsilon
    noise_std = math.sqrt(2) * laplace_scale
    return noise_std


def probabilistic_compute_utility_curve(a, b, k):

    """
    a: error-rate
    b: proba that error-rate will be respected
    n: total size of data requested by the user query
    k: number of computations/subqueries (aggregations: k-1)
    """
    # Special case for K = 1 because the computation:
    # 1-pow(1-b, 1) is not giving back precisely b (floating precision?)
    return a, 1 - math.pow(1 - b, 1 / k) if k > 1 else b
