from algorithms import *
import numpy as np
from visuals import *
from beta import *
import matplotlib.pyplot as plt
from scipy.stats import beta

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
    fig, (ax1, ax2) = plt.subplots(2, 1)

    y_mixture = beta.pdf(x_values, 2, 8) / 2 + beta.pdf(x_values, 8, 2) / 2
    ax2.plot(x_values, y_mixture, 'r-', lw=5, alpha=0.6, label='Beta mixture density')

    ax2.plot(x_values, y_values, 'b-', lw=5, alpha=0.6, label='Sample density')

    ax2.set(xlabel='x', ylabel='density',
        title= 'Density comparison')
    ax2.legend(loc="upper left")
    ax2.grid()

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

def plot_density_and_u(xis):
    
    fig, (ax1, ax2) = plt.subplots(nrows = 2, sharex = True)

    x_values = np.linspace(0, 1, 101)
    
    y_sample_density = np.zeros(101)
    I = quad(integrand, 0, 1, args = tuple(xis))[0]
    for i, x in enumerate(x_values):
        y_sample_density[i] = np.exp(u(x, xis)) / I

    y_mixture = beta.pdf(x_values, 2, 8) / 2 + beta.pdf(x_values, 8, 2) / 2
    ax2.plot(x_values, y_mixture, 'r-', lw=5, alpha=0.6, label='Beta mixture density')

    ax2.plot(x_values, y_sample_density, 'b-', lw=5, alpha=0.6, label='Sample density')

    ax2.set(xlabel='x', ylabel='density',
        title= 'Density comparison')
    ax2.legend(loc="upper left")
    ax2.grid()

    u_values = np.zeros(101)
    for i, x in enumerate(x_values):
        u_values[i] = u(x, xis)
    
    ax1.plot(x_values, u_values)

    ax1.set(ylabel='u(x)', title= 'u plot')
    ax1.grid()

    #fig.savefig("density_and_u.png")
    plt.show()

def beta_mix_analysis(folder):

    samples = np.load(folder + '/samples.npy')

    s = samples[-1]
    #log_density = get_log_target('beta_mix', 'inv_lap', 4)(s)

    #print(s)
    #print(log_density)

    #plot_u(s)
    #plot_density(s)
    plot_density_and_u(s)

def plot_beta_mixture():

    fig, ax = plt.subplots(1, 1)
    x_values = np.linspace(0, 1, 101)
    y_values = beta.pdf(x_values, 2, 8) / 2 + beta.pdf(x_values, 8, 2) / 2
    ax.plot(x_values, beta.pdf(x_values, 2, 8), 'r-', lw=5, alpha=0.6, label='Beta(2, 8)')
    ax.plot(x_values, beta.pdf(x_values, 8, 2), 'b-', lw=5, alpha=0.6, label='Beta(8, 2)')
    ax.plot(x_values, y_values, 'g-', lw=5, alpha=0.6, label='Beta mixture')
    ax.legend(loc="upper left")
    plt.show()


#folder = compute_beta_mcmc(tuned_pCN, 100, 5000)

#folder = "./output/tuned_pCN_beta_mix_inv_lap_dim20/run1"
#print(folder)
#beta_mix_analysis(folder)
plot_beta_mixture()