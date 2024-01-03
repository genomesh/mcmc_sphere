from algorithms import *
import numpy as np
from visuals import *
from beta import *
import matplotlib.pyplot as plt

def compute_beta_mcmc(algorithm, dy, num_samples):
    folder = algorithm(num_samples, dy, 'beta_mix', 'inv_lap')
    basic_plots(folder)
    mcmc_plots(folder)
    return folder

def plot_density(xis):
    x_values = np.linspace(0, 1, 101)
    y_values = np.zeros(101)
    I = quad(integrand, 0, 1, args = tuple(xis))[0]
    for i, x in enumerate(x_values):
        y_values[i] = np.exp(u(x, xis)) / I
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)

    ax.set(xlabel='x', ylabel='density',
        title= 'Density plot')
    ax.grid()

    #fig.savefig("density_plot.png")
    plt.show()

def plot_u(xis):
    x_values = np.linspace(0, 1, 101)
    y_values = np.zeros(101)
    for i, x in enumerate(x_values):
        y_values[i] = u(x, xis)
    
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)

    ax.set(xlabel='x', ylabel='u(x)',
        title= 'u plot')
    ax.grid()

    #fig.savefig("u_plot.png")
    plt.show()

def beta_mix_analysis(folder):

    samples = np.load(folder + '/samples.npy')

    s = samples[-1]
    #log_density = get_log_target('beta_mix', 'inv_lap', 4)(s)

    #print(s)
    #print(log_density)

    plot_u(s)
    plot_density(s)


#folder = "./output/rw_beta_mix_inv_lap_dim4/run6"
folder = compute_beta_mcmc(tuned_pCN, 9, 5000)
print(folder)
beta_mix_analysis(folder)