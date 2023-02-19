import math


def get_laplace_epsilon(a, b, n, k):
    if k == 1:
        epsilon = math.log(1 / b) / (n * a)
    elif k >= math.log(2 / b):
        epsilon = math.sqrt(k * 8 * math.log(2 / b)) / (n * a)
    else:
        epsilon = (math.log(2 / b) * math.sqrt(8)) / (n * a)
    return epsilon


def get_pmw_epsilon(alpha, beta, n, max_pmw_k):
    """
    Outputs the smallest epsilon that gives error at most alpha with proba at least 1- beta/max_pmw_k
    See "Per-query accuracy guarantees for aggregated PMWs" lemma.
    """
    return 4 * math.log(max_pmw_k / beta) / (alpha * n)
