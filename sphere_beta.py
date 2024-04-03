from algorithms import *
import numpy as np
from visuals import *
import matplotlib.pyplot as plt
import pandas as pd
from potentials import coal_potential, coal_data, normalise, beta_samples
from ess import update_correlation
from scipy.stats import beta

#folder = pCN(10000, 3, 'beta3', 'inv_lap', s = 0.5)

def beta_sphere_density(x, t):
    dimension = len(x)
    if np.linalg.norm(x) != 1:
        x = normalise(x)
    sum = 0
    for i in range(dimension):
        if i == 0:
            phi = 1
        else:
            phi = np.sqrt(2) * np.cos(np.pi * i * t)
        sum += x[i] * phi
    return sum ** 2

def plot_beta_sphere_density(x_array, dy):

    fig, ax = plt.subplots()

    t_values = np.linspace(0, 1, 501)
    y_values = np.zeros(501)

    for index, x in enumerate(x_array):
        
        for i, t in enumerate(t_values):
            y_values[i] = beta_sphere_density(x, t)
        
        if index == 0:
            ax.plot(t_values, y_values, color = 'black', alpha = 0.3, label = 'sample density')
        else:
            ax.plot(t_values, y_values, color = 'black', alpha = 0.3)

    for index, datapoint in enumerate(beta_samples[:dy]):
        print(datapoint)
        if index == 0:
            ax.plot(datapoint, 0, 'r.', markersize = 10, alpha = 1, label = 'datapoint')
        else:
            ax.plot(datapoint, 0, 'r.', markersize = 10, alpha = 1)


    y_mixture = beta.pdf(t_values, 2, 8) / 2 + beta.pdf(t_values, 8, 2) / 2
    ax.plot(t_values, y_mixture, 'r-', lw=2, alpha=1, label='True density')

    ax.set(xlabel='t', ylabel='density(t)')
    ax.grid()
    ax.legend()
    #fig.savefig("u_plot.png")
    plt.show()

def get_probability_less_half(x):
    x = normalise(x)
    dimension = len(x)
    sum = 0

    def sin_pi_over_two(k):
        if k % 2 == 0:
            return 0
        if k % 4 == 1:
            return 1
        if k % 4 == 3:
            return -1

    for i in range(dimension):
        for j in range(dimension):
            if i == 0 and j == 0:
                w = 1/2
            if i == 0 and j > 0:
                w = np.sqrt(2) * sin_pi_over_two(j) / (np.pi * j)
            if i > 0 and j == 0:
                w = np.sqrt(2) * sin_pi_over_two(i) / (np.pi * i)
            if i > 0 and i == j:
                w = 1/2
            if i > 0 and j > 0 and i != j:
                w = sin_pi_over_two(i+j) / (np.pi * (i+j)) + sin_pi_over_two(i-j) / (np.pi * (i-j))
            sum += x[i] * x[j] * w
    return sum

def get_prob_samples_less_half(folder):
    samples = np.load(folder + '/samples.npy')
    prob_samples = np.zeros(len(samples))
    for index, sample in enumerate(samples):
        if index % 500 == 0:
            print('Starting sample', index)
        prob_samples[index] = get_probability_less_half(sample)
    return prob_samples

def dim_vs_performance_beta():
    num_samples = 110000
    for algorithm in [tuned_pCN]: # [tuned_pCN, tuned_rw_mcmc, rw_mcmc, pCN]:
        for dim in [10]:
            for potential in ['beta3', 'beta50']:
                print('\nstarting dimension', dim, '\n')

                folder = algorithm(num_samples, dim, potential, 'inv_lap')
                print('\n', folder, '\n')

                samples = get_prob_samples_less_half(folder)[10000:]

                update_correlation(folder, samples)

example_folder_1 = './output/pCN_beta10_inv_lap_dim50/run1/samples.npy'

def plot_samples_densities(folder, samples_to_plot, dy):
    sample_values = np.load(folder)[10000:]
    plot_beta_sphere_density([sample_values[index] for index in samples_to_plot], dy)

folders_to_analyse = [
    'pCN_beta10_inv_lap_dim3',
    'pCN_beta10_inv_lap_dim5',
    'pCN_beta10_inv_lap_dim10',
    'pCN_beta10_inv_lap_dim20',
    'pCN_beta10_inv_lap_dim50',
    'pCN_beta10_inv_lap_dim100',
    'rw_beta10_inv_lap_dim3',
    'rw_beta10_inv_lap_dim5',
    'rw_beta10_inv_lap_dim10',
    'rw_beta10_inv_lap_dim20',
    'tuned_pCN_beta10_inv_lap_dim3',
    'tuned_pCN_beta10_inv_lap_dim5',
    'tuned_pCN_beta10_inv_lap_dim10',
    'tuned_pCN_beta10_inv_lap_dim20',
    'tuned_pCN_beta10_inv_lap_dim50',
    'tuned_pCN_beta10_inv_lap_dim100',
    'tuned_rw_beta10_inv_lap_dim3',
    'tuned_rw_beta10_inv_lap_dim5',
    'tuned_rw_beta10_inv_lap_dim10',
    'tuned_rw_beta10_inv_lap_dim20',
    'tuned_rw_beta10_inv_lap_dim50',
    'tuned_rw_beta10_inv_lap_dim100'
]

def get_average_jump_size(run_folder):
    jump_sizes = np.load(run_folder + '/jump_sizes.npy')
    return np.mean(jump_sizes)

def gather_metadata():

    data_to_plot = [
        'ess', 'autocorrelation', 'iact', 'mc_estimate', 'runtime (secs)',
        'empirical alpha', 'dimension', 'final_step_size', 'average jump size']
    
    algorithm_names = ['pCN','tuned_pCN','rw', 'tuned_rw']

    dict_to_plot = { algo : { key : [] for key in data_to_plot} for algo in algorithm_names }

    for folder in folders_to_analyse:

        with open('./output/' + folder + '/run1/metadata.json', 'r') as f:
            data = json.load(f)

            for key in data_to_plot:
                if key == 'final_step_size':
                    if data['tuning']:
                        dict_to_plot[data['algorithm']][key].append(data[key])
                    else:
                        dict_to_plot[data['algorithm']][key].append(data['step_size'])
                elif key == 'average jump size':
                    dict_to_plot[data['algorithm']][key].append(get_average_jump_size('./output/' + folder + '/run1'))
                else:
                    if key in data.keys():
                        dict_to_plot[data['algorithm']][key].append(data[key])
            
            
    return(dict_to_plot)

def plots_from_paper(metric, ylabel):

    dict_to_plot = gather_metadata()

    #fig, ax = plt.subplots()

    #ax.plot(dimension, iact, 'r_', markersize = 4)

    for algo in dict_to_plot.keys():
        if len(dict_to_plot[algo]['dimension']) == len(dict_to_plot[algo][metric]):
            plt.plot(dict_to_plot[algo]['dimension'], dict_to_plot[algo][metric], marker = 'o', markersize = 4, label = algo)

    plt.xlabel('dimension')
    plt.ylabel(ylabel)
    #plt.title('Integrated Autocorrelation time (IACT) vs dimension')
    plt.grid()
    plt.legend()
    #fig.savefig("u_plot.png")
    plt.show()

dy_analysis_folders = [
    './output/tuned_pCN_beta50_inv_lap_dim10/run1',
    './output/tuned_pCN_beta3_inv_lap_dim10/run1',
    './output/tuned_pCN_beta10_inv_lap_dim10/run1'
]

def plot_density_confidence_region(x_array, dy):

    fig, ax = plt.subplots()

    ax.set_ylim([0, 4])

    t_values = np.linspace(0, 1, 501)
    mean_values = np.zeros(501)
    percentile_95 = np.zeros(501)
    percentile_5 = np.zeros(501)

    #y_mixture = beta.pdf(t_values, 2, 8) / 2 + beta.pdf(t_values, 8, 2) / 2
    #ax.plot(t_values, y_mixture, color = 'red', lw=1, alpha=1, label='True density')

    density_values = np.zeros(len(x_array))

    for time_index, t in enumerate(t_values):
        for density_index, x in enumerate(x_array):
            density_values[density_index] = beta_sphere_density(x, t)
        mean_values[time_index] = np.mean(density_values)
        percentile_95[time_index] = np.percentile(density_values, 90)
        percentile_5[time_index] = np.percentile(density_values, 10)
    
    ax.plot(t_values, mean_values, color = 'black', alpha = 1, label = 'mean density')
    ax.plot(t_values, percentile_95, color = 'black', linestyle = 'dashed', alpha = 1, label = '90th percentile')
    ax.plot(t_values, percentile_5, color = 'black', linestyle = 'dashed', alpha = 1, label = '10th percentile')

    # y axis fixed 0-4
    # shade 80 percentile region?

    for datapoint in beta_samples[:dy]:
        ax.plot(datapoint, 0, 'r.', markersize = 10, alpha = 1)

    ax.set(xlabel='t', ylabel='density(t)',
        title= 'Density plot with ' + str(dy) + ' datapoints')
    
    ax.legend()

    ax.grid()

    #fig.savefig("u_plot.png")
    plt.show()

def plot_samples_confidence_regions(folder, samples_to_plot, dy):
    sample_values = np.load(folder)[10000:]
    plot_density_confidence_region([sample_values[index] for index in samples_to_plot], dy)

plot_samples_densities(example_folder_1, np.arange(100000)[::1000], 10)