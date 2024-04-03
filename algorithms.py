import numpy as np
from math import pi
from save_output import save_outputs
from targets import get_log_target, potentials, prior_cov_cholesky, prior_covs
from potentials import normalise
import time

def rw_mcmc(num_samples, dim, potential, prior_cov, s = 1):

    start_time = time.time()

    prop_cholesky = prior_cov_cholesky[prior_cov](dim)

    log_target_density = get_log_target(potential, prior_cov, dim)

    print_interval = int(max(num_samples / 100, 100))

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    jump_sizes = np.zeros(num_samples)
    accept_probs = np.zeros(num_samples)

    cur = np.full(dim, 10)
    samples[0] = cur
    cur_log_density = log_target_density(cur)

    for i in range(1, num_samples):
        prop = cur + s * prop_cholesky @ generator.standard_normal(size = dim)
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
            print(f"Sample {i}, accept rate: {emp_accept_prob:.2f}, log jumps: {np.log(emp_jump_size):.2f}, current log density: {cur_log_density:.2f}")

    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2),
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'second half emp alpha': np.mean(accept_probs[int(num_samples/2):]),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': False,
        'algorithm': 'rw',
        'prop_cov': 'identity',
        'dimension': dim,
        'step_size': s
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs)

    return run_folder

def tuned_rw_mcmc(num_samples, dim, potential, prior_cov):

    start_time = time.time()

    prop_cholesky = prior_cov_cholesky[prior_cov](dim)

    log_target_density = get_log_target(potential,  prior_cov, dim)

    print_interval = 500 #int(max(num_samples / 100, 100))

    step_size = 1
    final_tune = 10000

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
            emp_accept_prob = np.mean(accept_probs[i+1-print_interval:i])
            emp_jump_size = np.mean(jump_sizes[i+1-print_interval:i])
            print(f"Sample {i}, accept rate: {emp_accept_prob:.2f}, log jumps: {np.log(emp_jump_size):.2f}, current log density: {cur_log_density:.2f}")
            if i < final_tune:
                if emp_accept_prob > 0.5:
                    step_size *= 2
                    print(f'big tune step size up, new step size: {step_size:.2f}')
                elif emp_accept_prob > 0.28:
                    step_size *= 1.1
                    print(f'tune step size up, new step size: {step_size:.2f}')
                if emp_accept_prob < 0.1:
                    step_size *= 1/2
                    print(f'big tune step size down, new step size: {step_size:.2f}')
                elif emp_accept_prob < 0.18:
                    step_size *= 0.9
                    print(f'tune step size down, new step size: {step_size:.2f}')
                
    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2),
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': True,
        'algorithm': 'tuned_rw',
        'prop_cov': 'identity',
        'dimension': dim,
        'final_step_size': step_size
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs, step_sizes = step_sizes)

    return run_folder

def pCN(num_samples, dim, potential, prior_cov, s = 0.5):
    
    start_time = time.time()

    phi = potentials[potential]

    cov_cholesky = prior_cov_cholesky[prior_cov](dim)

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    accept_probs = np.zeros(num_samples)
    jump_sizes = np.zeros(num_samples)

    cur = np.full(dim, 10)
    samples[0] = cur
    cur_phi = phi(cur)

    print_interval = 500 #int(max(num_samples / 100, 100))

    for i in range(1, num_samples):
        w_k = generator.standard_normal(size = dim)
        prop = np.sqrt(1 - s*s) * cur + s * cov_cholesky @ w_k
        prop_phi = phi(prop)

        log_alpha = min(0, cur_phi - prop_phi)
        accept_probs[i] = np.exp(log_alpha)
        jump_sizes[i] = np.linalg.norm(prop - cur)

        u = generator.uniform(size = 1)

        if np.log(u) < log_alpha:
            cur = prop
            cur_phi = prop_phi

        samples[i] = cur
    
        if i % print_interval == 0:
            emp_accept_prob = np.mean(accept_probs[i+1-print_interval:i])
            emp_jump_size = np.mean(jump_sizes[i+1-print_interval:i])
            print(f"Sample {i}, accept rate: {emp_accept_prob:.2f}, log jumps: {np.log(emp_jump_size):.2f}, current potential: {cur_phi:.2f}")

    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2),
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': False,
        'algorithm': 'pCN',
        'dimension': dim,
        'step_size': s
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs)

    return run_folder

def tuned_pCN(num_samples, dim, potential, prior_cov, ):
    
    start_time = time.time()

    phi = potentials[potential]

    cov_cholesky = prior_cov_cholesky[prior_cov](dim)

    s = 1

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    accept_probs = np.zeros(num_samples)
    step_sizes = np.zeros(num_samples)
    jump_sizes = np.zeros(num_samples)

    cur = np.full(dim, 10)
    samples[0] = cur
    cur_phi = phi(cur)

    print_interval = 500 #int(max(num_samples / 100, 100))
    final_tune = 10 ** 4

    for i in range(1, num_samples):
        w_k = generator.standard_normal(size = dim)
        prop = np.sqrt(1 - s*s) * cur + s * cov_cholesky @ w_k
        prop_phi = phi(prop)

        log_alpha = min(0, cur_phi - prop_phi)
        accept_probs[i] = np.exp(log_alpha)
        jump_sizes[i] = np.linalg.norm(prop - cur)
        step_sizes[i] = s

        u = generator.uniform(size = 1)

        if np.log(u) < log_alpha:
            cur = prop
            cur_phi = prop_phi

        samples[i] = cur
    
        if i % print_interval == 0:
            emp_accept_prob = np.mean(accept_probs[i+1-print_interval:i])
            emp_jump_size = np.mean(jump_sizes[i+1-print_interval:i])
            print(f"Sample {i}, accept rate: {emp_accept_prob:.2f}, log jumps: {np.log(emp_jump_size):.2f}, current potential: {cur_phi:.2f}")
            if i < final_tune:
                if emp_accept_prob > 0.28 and s < 1:
                    s = min(1.1 * s, 1)
                    print(f'tune step size up, new step size: {s:.2f}')
                if emp_accept_prob < 0.18:
                    s *= 0.9
                    print(f'tune step size down, new step size: {s:.2f}')

    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2),
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': True,
        'algorithm': 'tuned_pCN',
        'dimension': dim,
        'final_step_size': s
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs, step_sizes = step_sizes)

    return run_folder

def pCN_sphere(num_samples, dim, potential, prior_cov, s = 0.5):
    
    start_time = time.time()

    phi = potentials[potential]

    cov_cholesky = prior_cov_cholesky[prior_cov](dim)
    cov = prior_covs[prior_cov](dim)

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    accept_probs = np.zeros(num_samples)
    jump_sizes = np.zeros(num_samples)

    cur = np.array([1] + [0] * (dim-1))
    samples[0] = cur
    cur_phi = phi(cur)

    print_interval = 500 #int(max(num_samples / 100, 100))

    for i in range(1, num_samples):
        r_2 = generator.gamma(shape = dim / 2, scale = 2 / (cur.T @ np.linalg.solve(cov, cur)))
        w_k = generator.standard_normal(size = dim)
        y_k = np.sqrt(1 - s*s) * np.sqrt(r_2) * cur + s * cov_cholesky @ w_k
        prop = normalise(y_k)
        prop_phi = phi(prop)

        log_alpha = min(0, cur_phi - prop_phi)
        accept_probs[i] = np.exp(log_alpha)
        jump_sizes[i] = np.linalg.norm(prop - cur)

        u = generator.uniform(size = 1)

        if np.log(u) < log_alpha:
            cur = prop
            cur_phi = prop_phi

        samples[i] = cur
    
        if i % print_interval == 0:
            emp_accept_prob = np.mean(accept_probs[i+1-print_interval:i])
            emp_jump_size = np.mean(jump_sizes[i+1-print_interval:i])
            print(f"Sample {i}, accept rate: {emp_accept_prob:.2f}, log jumps: {np.log(emp_jump_size):.2f}, current potential: {cur_phi:.2f}")

    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2),
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': False,
        'algorithm': 'pCN_sphere',
        'dimension': dim,
        'step_size': s
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs)

    return run_folder

def tuned_pCN_sphere(num_samples, dim, potential, prior_cov):
    
    start_time = time.time()

    phi = potentials[potential]

    cov_cholesky = prior_cov_cholesky[prior_cov](dim)
    cov = prior_covs[prior_cov](dim)

    s = 1

    generator = np.random.default_rng()

    samples = np.zeros((num_samples, dim))
    accept_probs = np.zeros(num_samples)
    step_sizes = np.zeros(num_samples)
    jump_sizes = np.zeros(num_samples)

    cur = np.array([1] + [0] * (dim-1))
    samples[0] = cur
    cur_phi = phi(cur)

    print_interval = 500 #int(max(num_samples / 100, 100))
    final_tune = 10 ** 4

    for i in range(1, num_samples):
        r_2 = generator.gamma(shape = dim / 2, scale = 2 / (cur.T @ np.linalg.solve(cov, cur)))
        w_k = generator.standard_normal(size = dim)
        y_k = np.sqrt(1 - s*s) * np.sqrt(r_2) * cur + s * cov_cholesky @ w_k
        prop = normalise(y_k)
        prop_phi = phi(prop)

        log_alpha = min(0, cur_phi - prop_phi)
        accept_probs[i] = np.exp(log_alpha)
        jump_sizes[i] = np.linalg.norm(prop - cur)
        step_sizes[i] = s

        u = generator.uniform(size = 1)

        if np.log(u) < log_alpha:
            cur = prop
            cur_phi = prop_phi

        samples[i] = cur
    
        if i % print_interval == 0:
            emp_accept_prob = np.mean(accept_probs[i+1-print_interval:i])
            emp_jump_size = np.mean(jump_sizes[i+1-print_interval:i])
            print(f"Sample {i}, accept rate: {emp_accept_prob:.2f}, log jumps: {np.log(emp_jump_size):.2f}, current potential: {cur_phi:.2f}")
            if i < final_tune:
                if emp_accept_prob > 0.28 and s < 1:
                    s = min(1.1 * s, 1)
                    print(f'tune step size up, new step size: {s:.2f}')
                if emp_accept_prob < 0.18:
                    s *= 0.9
                    print(f'tune step size down, new step size: {s:.2f}')

    meta_data = {
        'runtime (secs)': round(time.time() - start_time, 2),
        'number of samples': num_samples,
        'empirical alpha': np.mean(accept_probs),
        'potential': potential,
        'prior_cov': prior_cov,
        'tuning': True,
        'algorithm': 'tuned_pCN_sphere',
        'dimension': dim,
        'final_step_size': s
    }

    run_folder = save_outputs(meta_data, samples = samples, jump_sizes = jump_sizes, accept_probs = accept_probs, step_sizes = step_sizes)

    return run_folder
