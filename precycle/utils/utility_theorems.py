import math

import numpy as np


def get_laplace_epsilon(a, b, n, k):
    if k == 1:
        epsilon = math.log(1 / b) / (n * a)
    elif k >= math.log(2 / b):
        # Concentration branch
        epsilon = math.sqrt(k * 8 * math.log(2 / b)) / (n * a)
    else:
        # b_M branch
        epsilon = (math.log(2 / b) * math.sqrt(8)) / (n * a)
    return epsilon


def get_sv_epsilon(alpha, beta, n, l=1 / 2):
    """
    l=1/2 for SV threshold = alpha/2.
    Outputs only the beta_SV from Overleaf.
    You need to do a union bound with beta_Laplace to get a global beta = beta_SV + beta_Laplace
    that holds even for hard queries.
    """
    binary_eps = binary_search_epsilon(
        alpha=alpha, beta=beta, n=n, l=l, beta_tolerance=1e-5, extra_laplace=False
    )
    real_beta = sum_laplace_beta(binary_eps, n, alpha, l, extra_laplace=False)

    # Make sure that we didn't accidentatlly overspend budget
    assert real_beta < beta

    return binary_eps


def get_pmw_epsilon(alpha, beta, n):
    l = 1 / 2  # We can't change l in the vanilla PMW
    binary_eps = binary_search_epsilon(
        alpha=alpha, beta=beta, n=n, l=l, beta_tolerance=1e-5, extra_laplace=True
    )
    real_beta = sum_laplace_beta(binary_eps, n, alpha, l, extra_laplace=True)

    # Make sure that we didn't accidentatlly overspend budget
    assert real_beta < beta

    return binary_eps


def get_pmw_epsilon_loose(alpha, beta, n, max_pmw_k):
    """
    Outputs the smallest epsilon that gives error at most alpha with proba at least 1- beta/max_pmw_k
    See "Per-query accuracy guarantees for aggregated PMWs" lemma.
    """
    return 4 * math.log(max_pmw_k / beta) / (alpha * n)


def loose_epsilon(alpha, beta, n, l):
    return 2 * np.log(1 / beta) / (l * alpha * n)


def sum_laplace_beta(epsilon, n, alpha, l=1 / 2, extra_laplace=False):
    """
    If extra_laplace = True, we add an extra failure probability coming from the hard query Laplace noise
    See "Concentrated per-query accuracy guarantees for a single PMW" lemma.
    """
    e = l * alpha * n * epsilon
    beta_sv = (1 / 2 + e / 4) * np.exp(-e)
    beta_laplace = np.exp(-2 * e) if extra_laplace else 0
    return beta_laplace + beta_sv


def binary_search_epsilon(alpha, beta, n, l, beta_tolerance=1e-5, extra_laplace=False):
    """
    Find the lowest epsilon that satisfies the failure probability guarantee.
    If extra_laplace = True, this is for a full PMW. Otherwise, it's just for a single SV.
    """
    eps_low = 0
    eps_high = loose_epsilon(alpha, beta, n, l)
    # Make sure that the initial upper bound is large enough
    assert sum_laplace_beta(eps_high, n, alpha, l=l, extra_laplace=extra_laplace) < beta

    real_beta = 0

    # Bring real_beta close to beta, but from below (conservative)
    while real_beta < beta - beta_tolerance:
        eps_mid = (eps_low + eps_high) / 2
        beta_mid = sum_laplace_beta(eps_mid, n, alpha, l=l, extra_laplace=extra_laplace)
        # print(
        #     f"{eps_low} < {eps_mid} < {eps_high} gives beta={beta_mid}. Target {beta}"
        # )

        if beta_mid < beta:
            eps_high = eps_mid
            real_beta = beta_mid
        else:
            # Don't update the real_beta, you can only exit the loop if real_beta < beta - beta_tolerance
            eps_low = eps_mid

    return eps_high
