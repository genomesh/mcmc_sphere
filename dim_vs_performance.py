import numpy as np
from algorithms import *
import json
import matplotlib.pyplot as plt

def step_vs_dim_computation(algorithm, num_samples, potential, prior_cov):
    # for algorithms that tune step size.
    dims = [1, 3, 5, 8, 13, 20, 30]

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
    plt.title(metadata["algorithm"])
    plt.show()

def step_vs_dim_analysis():

    folders = step_vs_dim_computation(tuned_rw_mcmc, 10000, 'phi2', 'id')
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
    dims = [1, 3, 5, 8, 13, 20]

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
        #empirical_accept_probs.append(metadata['second half emp alpha'])
        empirical_accept_probs.append(metadata["empirical alpha"])
        # need to use some burn in
    
    plt.scatter(np.log(dims), empirical_accept_probs)
    plt.xlabel('log of dimension')
    plt.ylabel('empirical acceptance probability')
    plt.title(metadata["algorithm"])
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

#folders = accept_vs_dim_computation(rw_mcmc, 5000, 'beta_mix', 'inv_lap')
#print(folders)
#print(step_vs_dim_computation(rw_mcmc_tuning, 25000, 'phi2', 'id'))
'''
#tuned rw
folders = [
    './output/tuned_rw_beta_mix_inv_lap_dim1/run1',
    './output/tuned_rw_beta_mix_inv_lap_dim3/run1',
    './output/tuned_rw_beta_mix_inv_lap_dim5/run1',
    './output/tuned_rw_beta_mix_inv_lap_dim8/run1',
    './output/tuned_rw_beta_mix_inv_lap_dim13/run1',
    './output/tuned_rw_beta_mix_inv_lap_dim20/run1',
    './output/tuned_rw_beta_mix_inv_lap_dim30/run1'
]
'''

folders = ['./output/pCN_beta_mix_inv_lap_dim1/run1', './output/pCN_beta_mix_inv_lap_dim3/run1', './output/pCN_beta_mix_inv_lap_dim5/run1', './output/pCN_beta_mix_inv_lap_dim8/run1', './output/pCN_beta_mix_inv_lap_dim13/run1', './output/pCN_beta_mix_inv_lap_dim20/run1', './output/pCN_beta_mix_inv_lap_dim30/run1']
#['./output/rw_beta_mix_inv_lap_dim1/run1', './output/rw_beta_mix_inv_lap_dim3/run1', './output/rw_beta_mix_inv_lap_dim5/run1', './output/rw_beta_mix_inv_lap_dim8/run2', './output/rw_beta_mix_inv_lap_dim13/run1', './output/rw_beta_mix_inv_lap_dim20/run1', './output/rw_beta_mix_inv_lap_dim30/run1']
accept_vs_dim_visual(folders)