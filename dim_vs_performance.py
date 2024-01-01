import numpy as np
from algorithms import *
import json
import matplotlib.pyplot as plt

def step_vs_dim_computation(algorithm, num_samples, potential, prior_cov):
    # for algorithms that tune step size.
    dims = [1000] #[3, 10, 20, 50, 100, 200, 500, 700, 1000]

    folders = []

    for dim in dims:
        print(f'\n starting dimension {dim} \n')
        folders.append(algorithm(num_samples, dim, potential, prior_cov))

    return folders

def step_vs_dim_visual(folders):
    final_step_sizes = []
    dims = []
    for folder in folders:
        with open(folder + '/metadata.json', 'r') as j:
            metadata = json.loads(j.read())
        dims.append(metadata["dimension"])
        final_step_sizes.append(metadata["final_step_size"])
    
    plt.scatter(np.log(dims), final_step_sizes)
    plt.xlabel('log of dimension')
    plt.ylabel('final step size')
    plt.show()

def step_vs_dim_analysis():

    #folders = step_vs_dim_computation(rw_mcmc_tuning, 10000, 'phi2', 'id')
    #print(folders)
    #print(step_vs_dim_computation(rw_mcmc_tuning, 25000, 'phi2', 'id'))

    folders = [
        './output/tuned_rw_phi2_id_dim3/run1',
        './output/tuned_rw_phi2_id_dim10/run1',
        './output/tuned_rw_phi2_id_dim20/run1',
        './output/tuned_rw_phi2_id_dim50/run1',
        './output/tuned_rw_phi2_id_dim100/run1',
        './output/tuned_rw_phi2_id_dim200/run1',
        './output/tuned_rw_phi2_id_dim500/run1',
        './output/tuned_rw_phi2_id_dim700/run1',
        './output/tuned_rw_phi2_id_dim1000/run2'
    ]

    step_vs_dim_visual(folders)

def accept_vs_dim_computation(algorithm, num_samples, potential, prior_cov):
    # for algorithms that don't tune step size.
    dims = [3, 10, 20, 50, 100, 200, 500] #, 700, 1000, 1500]

    folders = []

    for dim in dims:
        print(f'\n starting dimension {dim} \n')
        folders.append(algorithm(num_samples, dim, potential, prior_cov, s=0.25))

    return folders

def accept_vs_dim_visual(folders):

    empirical_accept_probs = []
    dims = []
    for folder in folders:
        with open(folder + '/metadata.json', 'r') as j:
            metadata = json.loads(j.read())
        dims.append(metadata["dimension"])
        empirical_accept_probs.append(metadata['second half emp alpha'])
        # need to use some burn in
    
    plt.scatter(np.log(dims), empirical_accept_probs)
    plt.xlabel('log of dimension')
    plt.ylabel('empirical acceptance probability')
    plt.show()

def accept_vs_dim_analysis():
        
    #folders = accept_vs_dim_computation(rw_mcmc, 10000, 'phi2', 'id')
    #print(folders)
    #print(accept_vs_dim_computation(rw_mcmc_tuning, 25000, 'phi2', 'id'))

    folders = [
        './output/rw_phi2_id_dim3/run2',
        './output/rw_phi2_id_dim10/run2',
        './output/rw_phi2_id_dim20/run2',
        './output/rw_phi2_id_dim50/run2',
        './output/rw_phi2_id_dim100/run2',
        './output/rw_phi2_id_dim200/run2',
        './output/rw_phi2_id_dim500/run2'
    ]

    accept_vs_dim_visual(folders)
