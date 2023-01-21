import math


# def compute_utility_curve(utility, utility_beta, n, k):
#     # TODO: replace by generic std?
#     def min_epsilon(u, b, n, k):
#         return (math.sqrt(8 * k) * math.log(2 / b)) / (n * u)
#     return min_epsilon(utility, utility_beta, n, k)


def compute_utility_curve(utility, utility_beta, n, k):

    """
    utility: error-rate
    utility_beta: proba that error-rate will be respected
    n: total size of data requested by the user query
    k: number of computations/subqueries (aggregations: k-1)
    """

    def min_epsilon(u, b, n, k):
        """Returns the minimum epsilon that needs to be spent per computation if we have k aggregations"""

        if k >= math.log(2 / b):
            epsilon = math.sqrt(k * 8 * math.log(2 / b)) / (n * u)
        else:
            epsilon = (math.log(2 / b) * math.sqrt(8)) / (n * u)
        return epsilon

    return min_epsilon(utility, utility_beta, n, k)
