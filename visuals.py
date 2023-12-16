from algorithms import *
import seaborn as sns
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
import json
import numpy as np


def basic_plots(directory):

    #output = rw_mcmc(num_samples, dim, log_target_density, tuning_flag)
    #need to replace with reading file, json and npy

    with open(directory + '/metadata.json', 'r') as j:
        metadata = json.loads(j.read())
    
    samples = np.load(directory + '/samples.npy')
    accept_probs = np.load(directory + '/accept_probs.npy')
    jump_sizes = np.load(directory + '/jump_sizes.npy')

    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(x, y, 'x')
    ax1.set_title('First and second coordinates')
    ax1.axis('equal')
    ax2.plot(accept_probs, 'x')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_title('Acceptance probabilities')
    #ax3.plot(jump_sizes, 'x')
    #ax3.set_title('Jump Sizes')
    counts, bins = np.histogram(x)
    ax3.stairs(counts, bins)
    ax3.set_title('Third coordinate histogram')
    ax4.plot(x)
    ax4.set_title('Third coordinate trace')
    plt.show()

def mcmc_plots(directory):
    with open(directory + '/metadata.json', 'r') as j:
        metadata = json.loads(j.read())
    
    samples = np.load(directory + '/samples.npy')
    accept_probs = np.load(directory + '/accept_probs.npy')
    jump_sizes = np.load(directory + '/jump_sizes.npy')

    if metadata['dimension'] < 4:
        return False
    
    alpha = min(1, 3 / np.sqrt(metadata['number of samples']))

    df = pd.DataFrame(samples)[[0,1,2,3]]
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15, alpha = alpha)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.show()

#test_pCN(phi = phi_two)

#basic_plots('./output/rw_phi1_id_dim3/run2')