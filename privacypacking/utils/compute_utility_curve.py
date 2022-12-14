import math


def compute_utility_curve(utility, delta, n, epsilon_threshold):
    def utility_curve(k, delta, u):
        return (math.sqrt(8 * k) * math.log(2 / delta)) / u

    f = {}
    for k in range(1, n + 1):
        e = utility_curve(k, delta, utility)
        if e > epsilon_threshold:
            break
        f[k] = e
    return f
