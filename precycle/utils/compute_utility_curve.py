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

    if k == 1:
        epsilon = math.log(1 / b) / (n * a)
    elif k >= math.log(2 / b):
        epsilon = math.sqrt(k * 8 * math.log(2 / b)) / (n * a)
    else:
        epsilon = (math.log(2 / b) * math.sqrt(8)) / (n * a)

    sensitivity = 1 / n_i
    laplace_scale = sensitivity / epsilon
    noise_std = math.sqrt(2) * laplace_scale
    return noise_std


def get_pmw_epsilon(alpha, beta, n, max_pmw_k):
    """
    Outputs the smallest epsilon that gives error at most alpha with proba at least 1- beta/max_pmw_k
    See "Per-query accuracy guarantees for aggregated PMWs" lemma.
    """
    return 4 * math.log(max_pmw_k / beta) / (alpha * n)


def probabilistic_compute_utility_curve(a, b, n_i, k):

    """
    a: error-rate
    b: proba that error-rate will be respected
    n: total size of data requested by the user query
    k: number of computations/subqueries (aggregations: k-1)
    """
    if k == 1:
        # Actually I'm not sure this one holds, the concentration lemma works only under some conditions for alpha
        epsilon = math.log(2 / b) / (a * n_i)
    else:
        epsilon = (math.log((3 * k) / b) * 8) / (a * (n_i / k))

    return a, 1 / epsilon
