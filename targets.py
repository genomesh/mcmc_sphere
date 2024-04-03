import numpy as np
from beta import beta_mixture_potential
from potentials import coal_potential, generate_beta_sphere_potential

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
    alpha = 2
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
    exponent = diff.T @ np.linalg.solve(cov, diff) / 2
    return (- exponent)

def standard_normal_log_density(dim):
    return lambda x: np.linalg.norm(x) / 2

potentials = {
    'phi1': phi_one,
    'phi2': phi_two,
    'phi3': phi_three,
    'beta_mix': beta_mixture_potential,
    'coal': coal_potential,
    'beta3': generate_beta_sphere_potential(3),
    'beta10': generate_beta_sphere_potential(10),
    'beta50': generate_beta_sphere_potential(50)
}

prior_covs = {
    'id': lambda dim: np.eye(dim),
    'inv_lap': lambda dy: np.diag([1/((n+1)**2) for n in range(dy)]),
    'matern': lambda dy: np.diag([0.25 / (0.1 + (np.pi * i) ** 2) for i in range(dy)])
}

prior_cov_cholesky = {
    'id': lambda dim: np.eye(dim),
    'inv_lap': lambda dy: np.diag([1/(n+1) for n in range(dy)]),
    'matern': lambda dy: np.diag([0.5 / np.sqrt(0.1 + (np.pi * i) ** 2) for i in range(dy)])
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