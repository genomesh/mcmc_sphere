import numpy as np
from functools import partial

# Note phi functions correspond to potential.
# So, up to proportionality,
# p_target(x) = e ^ (- phi(x)) * Normal(x; 0, prior_cov)

def phi_one(x):
    y = x[0]
    if y > 1 and y < 2:
        return 0
    if y < -1 and y > -2:
        return 0
    return 2

def phi_two(x):
    alpha = 1
    y = x[0] + x[1]
    num = (y ** 2 - alpha) ** 2
    y = x[2]
    num += (y ** 2 - alpha) ** 2
    return(num)

def phi_three(x):
    y = x[0]
    num = abs(y) ** (3 / 2) - 15 * abs(y) ** (3 / 4)
    return(num/4)

def get_normal_log_density(x, mean, cov):
    diff = x - mean
    exponent = diff.T @ np.linalg.solve(cov, diff)
    return (- exponent)

def standard_normal_log_density(dim):
    cov = np.identity(dim)
    mean = np.zeros(dim)
    return partial(get_normal_log_density, mean = mean, cov = cov)

potentials = {
    'phi1': phi_one,
    'phi2': phi_two,
    'phi3': phi_three
}

prior_covs = {
    'id': lambda dim: np.eye(dim),
    'inv_lap': lambda dim: np.eye(dim) # to implement
}

def get_log_target(potential, prior_cov, dim):
    # This might be very inefficient.
    # Maybe should swap from functional to object-oriented?
    def log_target(x):
        pot = potentials[potential](x)
        cov = prior_covs[prior_cov](dim)
        normal_exp = get_normal_log_density(x, mean = np.zeros(dim), cov = cov)
        return normal_exp - pot
    return log_target