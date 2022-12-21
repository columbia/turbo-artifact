import math


def compute_utility_curve(utility, p, n, epsilon_threshold=None):
    # This assumes e-dp (delta=0)
    def utility_curve(k, p, u):
        return (math.sqrt(8 * k) * math.log(2 / p)) / u

    f = {}
    for k in range(1, n + 1):
        e = utility_curve(k, p, utility)
        if epsilon_threshold is not None and e > epsilon_threshold:
            break
        f[k] = e
    return f
