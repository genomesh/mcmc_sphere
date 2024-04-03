from algorithms import *
import numpy as np
from visuals import *
import matplotlib.pyplot as plt
import pandas as pd
from potentials import coal_potential, coal_data, normalise
from ess import update_correlation

def run_coal_mcmc(algorithm, dimension, num_samples):
    folder = algorithm(num_samples, dimension, 'coal', 'matern')
    #basic_plots(folder)
    #mcmc_plots(folder)
    return folder

def coal_density(x, t):
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
    
def plot_coal_density(x_array):

    fig, ax = plt.subplots()

    for index, x in enumerate(x_array):
        t_values = np.linspace(0, 1, 101)
        y_values = np.zeros(101)
        for i, t in enumerate(t_values):
            y_values[i] = coal_density(x, t)
        
        if index == 0:
            ax.plot(t_values, y_values, color = 'black', alpha = 0.2, label = 'sample density')
        else:
            ax.plot(t_values, y_values, color = 'black', alpha = 0.2)



    for idx, row in coal_data.iterrows():
        if idx == 0:
            ax.plot(row['scaled_year'], 0, 'r.', markersize = 4 * row['Count'], alpha = 0.8, label = 'datapoint')
        elif row['Count'] > 0:
            ax.plot(row['scaled_year'], 0, 'r.', markersize = 4 * row['Count'], alpha = 0.8)

    ax.set(xlabel='t', ylabel='density(t)')
    ax.grid()

    ax.legend()
    #fig.savefig("u_plot.png")
    plt.show()

# find the probability of a coal disaster between 1900 and 1916.
def get_probability_disaster(x):
    x = normalise(x)
    dimension = len(x)
    sum = 0
    b = 0.574
    a = 0.435
    for i in range(dimension):
        for j in range(dimension):
            if i == 0 and j == 0:
                w = b - a
            if i == 0 and j > 0:
                w = np.sqrt(2) * (np.sin(np.pi * j * b) - np.sin(np.pi * j * a)) / (np.pi * j)
            if i > 0 and j == 0:
                w = np.sqrt(2) * (np.sin(np.pi * i * b) - np.sin(np.pi * i * a)) / (np.pi * i)
            if i > 0 and i == j:
                w = b - a + (np.sin(2 * np.pi * i * b) - np.sin(2 * np.pi * i * a)) / (2 * np.pi * i)
            if i > 0 and j > 0 and i != j:
                w = (np.sin(np.pi * (i+j) * b) - np.sin(np.pi * (i+j) * a)) / (np.pi * (i+j)) + (
                    (np.sin(np.pi * (i-j) * b) - np.sin(np.pi * (i-j) * a)) / (np.pi * (i-j)) )
            sum += x[i] * x[j] * w
    return sum

def get_prob_samples(folder):
    samples = np.load(folder + '/samples.npy')
    prob_samples = np.zeros(len(samples))
    for index, sample in enumerate(samples):
        if index % 500 == 0:
            print('Starting sample', index)
        prob_samples[index] = get_probability_disaster(sample)
    return prob_samples

def get_average_jump_size(run_folder):
    jump_sizes = np.load(run_folder + '/jump_sizes.npy')
    return np.mean(jump_sizes)

def paper_analysis():
    num_samples = 110000
    for dim in [10, 20, 30, 40, 50]:
        print('\nstarting dimension', dim, '\n')

        folder = pCN(num_samples, dim, 'coal', 'matern', s = 0.2)
        print('\n', folder, '\n')

        samples = get_prob_samples(folder)[10000:]

        update_correlation(folder, samples)

folders_to_analyse = [
    'pCN_sphere_coal_matern_dim10', 'pCN_sphere_coal_matern_dim20', 'pCN_sphere_coal_matern_dim30', 'pCN_sphere_coal_matern_dim40', 'pCN_sphere_coal_matern_dim50', 'pCN_sphere_coal_matern_dim100',
    'tuned_pCN_coal_matern_dim10', 'tuned_pCN_coal_matern_dim20', 'tuned_pCN_coal_matern_dim30', 'tuned_pCN_coal_matern_dim40', 'tuned_pCN_coal_matern_dim50', 'tuned_pCN_coal_matern_dim100',
    'pCN_coal_matern_dim10', 'pCN_coal_matern_dim20',
    'tuned_rw_coal_matern_dim10', 'tuned_rw_coal_matern_dim20', 'tuned_rw_coal_matern_dim30', 'tuned_rw_coal_matern_dim40', 'tuned_rw_coal_matern_dim50']

def gather_metadata():

    data_to_plot = [
        'ess', 'autocorrelation', 'iact', 'mc_estimate', 'runtime (secs)',
        'empirical alpha', 'dimension', 'final_step_size', 'average jump size']
    
    algorithm_names = ['pCN_sphere','tuned_pCN','pCN', 'tuned_rw']

    #info_for_algo = 

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

#plots_from_paper('iact', 'IACT')

def plot_density_confidence_region(x_array):

    fig, ax = plt.subplots()

    t_values = np.linspace(0, 1, 501)
    mean_values = np.zeros(501)
    percentile_95 = np.zeros(501)
    percentile_5 = np.zeros(501)

    #y_mixture = beta.pdf(t_values, 2, 8) / 2 + beta.pdf(t_values, 8, 2) / 2
    #ax.plot(t_values, y_mixture, color = 'red', lw=1, alpha=1, label='True density')

    density_values = np.zeros(len(x_array))

    for time_index, t in enumerate(t_values):
        for density_index, x in enumerate(x_array):
            density_values[density_index] = coal_density(x, t)
        mean_values[time_index] = np.mean(density_values)
        percentile_95[time_index] = np.percentile(density_values, 90)
        percentile_5[time_index] = np.percentile(density_values, 10)
    
    ax.plot(t_values, mean_values, color = 'red', alpha = 1, label = 'mean density')
    ax.plot(t_values, percentile_95, color = 'blue', linestyle = 'dashed', alpha = 1, label = '90th percentile')
    ax.plot(t_values, percentile_5, color = 'blue', linestyle = 'dashed', alpha = 1, label = '10th percentile')

    # y axis fixed 0-4
    # shade 80 percentile region?

    for idx, row in coal_data.iterrows():
        if idx == 0:
            ax.plot(row['scaled_year'], 0, 'r.', markersize = 4 * row['Count'], alpha = 0.8, label = 'datapoint')
        elif row['Count'] > 0:
            ax.plot(row['scaled_year'], 0, 'r.', markersize = 4 * row['Count'], alpha = 0.8)

    ax.set(xlabel='t', ylabel='density(t)') #, title= 'Density plot with datapoints'
    
    ax.legend()

    ax.grid()

    #fig.savefig("u_plot.png")
    plt.show()

def plot_samples_confidence_regions(folder, samples_to_plot):
    sample_values = np.load(folder)[10000:]
    plot_density_confidence_region([sample_values[index] for index in samples_to_plot])

#plot_samples_confidence_regions('./output/pCN_sphere_coal_matern_dim100/run1/samples.npy', np.arange(100000)[::100])
    
def plot_samples_densities(folder, samples_to_plot):
    sample_values = np.load(folder)[10000:]
    plot_coal_density([sample_values[index] for index in samples_to_plot])

#plot_samples_confidence_regions('./output/pCN_sphere_coal_matern_dim100/run1/samples.npy', np.arange(100000)[::100])
plot_samples_densities('./output/pCN_sphere_coal_matern_dim100/run1/samples.npy', np.arange(100000)[::500])