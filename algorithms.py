import numpy as np
from math import pi
from save_output import save_outputs
from targets import get_log_target, potentials, prior_covs
import time


def rw_mcmc(num_samples, dim, potential, prior_cov):

    start_time = time.time()

    prop_cov = np.eye(dim)
    prop_cholesky = np.linalg.cholesky(prop_cov)

    log_target_density = get_log_target(potential,  prior_cov, dim)

    print_interval = int(max(num_samples / 100, 100))

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    jump_sizes = np.zeros(num_samples)
    accept_probs = np.zeros(num_samples)

    cur = np.full(dim, 10)
    samples[0] = cur
    cur_log_density = log_target_density(cur)

    for i in range(1, num_samples):
        prop = cur + prop_cholesky @ generator.standard_normal(size = dim)
        prop_log_density = log_target_density(prop)

        log_alpha = min(0, prop_log_density - cur_log_density)

        accept_probs[i] = np.exp(log_alpha)
        jump_sizes[i] = np.linalg.norm(prop - cur)

        u = generator.uniform(size = 1)

        if np.log(u) < log_alpha:
            cur = prop
            cur_log_density = prop_log_density

        samples[i] = cur

        if i % print_interval == 0:
            emp_accept_prob = np.mean(accept_probs[i+1-print_interval:i])
            emp_jump_size = np.mean(jump_sizes[i+1-print_interval:i])
            print(f"Sample {i}, acceptance rate: {emp_accept_prob:.2f}, log jump size: {np.log(emp_jump_size):.2f}")

    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2), # to add.
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': False,
        'algorithm': 'rw',
        'prop_cov': 'identity',
        'dimension': dim
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs)

    return run_folder

def rw_mcmc_tuning(num_samples, dim, potential, prior_cov):
    
    start_time = time.time()

    prop_cov = np.eye(dim)
    prop_cholesky = np.linalg.cholesky(prop_cov)

    log_target_density = get_log_target(potential,  prior_cov, dim)

    print_interval = int(max(num_samples / 100, 100))

    step_size = 1
    final_tune = num_samples
    recent_tune = 0

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    step_sizes = np.zeros(num_samples)
    jump_sizes = np.zeros(num_samples)
    accept_probs = np.zeros(num_samples)

    cur = np.full(dim, 10)
    samples[0] = cur
    cur_log_density = log_target_density(cur)

    for i in range(1, num_samples):
        prop = cur + step_size * prop_cholesky @ generator.standard_normal(size = dim)
        prop_log_density = log_target_density(prop)

        log_alpha = min(0, prop_log_density - cur_log_density)

        step_sizes[i] = step_size
        accept_probs[i] = np.exp(log_alpha)
        jump_sizes[i] = np.linalg.norm(prop - cur)

        u = generator.uniform(size = 1)

        if np.log(u) < log_alpha:
            cur = prop
            cur_log_density = prop_log_density

        samples[i] = cur

        if i % print_interval == 0:
            window_start = max(i-500, 0) + 1
            emp_accept_prob = np.mean(accept_probs[window_start:i])
            emp_jump_size = np.mean(jump_sizes[i+1-print_interval:i])
            print(f"Sample {i}, acceptance rate: {emp_accept_prob:.2f}, log jump size: {np.log(emp_jump_size):.2f}")
            if i < final_tune:
                if emp_accept_prob > 0.5:
                    print('tune step size up')
                    step_size *= 2
                    recent_tune = i
                elif emp_accept_prob > 0.28:
                    print('tune step size up')
                    step_size *= 1.1
                    recent_tune = i
                if emp_accept_prob < 0.1:
                    print('tune step size down')
                    step_size *= 1/2
                    recent_tune = i
                elif emp_accept_prob < 0.18:
                    print('tune step size down')
                    step_size *= 0.9
                    recent_tune = i
                
    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2), # to add.
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': False,
        'algorithm': 'tuned_rw',
        'prop_cov': 'identity',
        'dimension': dim,
        'final_step_size': step_sizes[-1]
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs, step_sizes = step_sizes)

    return run_folder

def pCN(num_samples, dim, phi, s = 0.5):
    prior_cov = np.eye(dim)
    cov_cholesky = np.linalg.cholesky(prior_cov)

    cur = np.full(dim, 3)

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    samples[0] = cur
    cur_phi = phi(cur) # * log_likelihood(cur)

    accept_probs = np.zeros(num_samples)
    step_sizes = np.zeros(num_samples)

    for i in range(1, num_samples):
        w_k = generator.standard_normal(size = dim)
        prop = np.sqrt(1 - s*s) * cur + s * cov_cholesky @ w_k
        step_sizes[i] = np.linalg.norm(prop - cur)
        prop_phi = phi(prop)

        log_alpha = min(0, cur_phi - prop_phi)
        accept_probs[i] = np.exp(log_alpha)

        u = generator.uniform(size = 1)

        if np.log(u) < log_alpha:
            cur = prop
            cur_phi = prop_phi

        samples[i] = cur
    
        if i % 500 == 0:
            window_start = max(i-500, 0) + 1
            emp_accept_prob = np.mean(accept_probs[window_start:i])
            print(f"Sample {i}, acceptance rate: {emp_accept_prob:.2f}")

    return {
        'samples': samples,
        'accept_probs': accept_probs,
        'step_sizes': step_sizes
    }
