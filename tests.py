import numpy as np
from algorithms import *

def step_size_fall_off(log_target_density):
    num_samples = 10000
    dims = [1, 3, 10, 20, 50, 100, 200, 500] #700, 1000, 1500]

    final_step_sizes = np.zeros(len(dims))
    for i, dim in enumerate(dims):
        print(f'\n starting dimension {dim} \n')
        cov = np.identity(dim)
        mean = np.zeros(dim)
        #partial(get_normal_log_density, mean = mean, cov = cov)
        output = rw_mcmc(num_samples, dim, log_target_density(dim), tuning_flag = True)
        final_step_sizes[i] = output['final_step_size']
    # separate testing and vis.
    plt.scatter(np.log(dims), final_step_sizes)
    plt.xlabel('log of dimension')
    plt.ylabel('final step size')
    plt.show()


def accept_fall_off():
    tuning_flag = False
    num_samples = 10000
    
    dims = [1, 3, 10, 20, 50, 100, 200, 500] #, 700, 1000, 1500]
    final_emp_accept = np.zeros(len(dims))
    for i, dim in enumerate(dims):
        print(f'\n starting dimension {dim} \n')
        cov = np.identity(dim)
        mean = np.zeros(dim)
        log_target_density = partial(get_normal_log_density, mean = mean, cov = cov)
        output = rw_mcmc(num_samples, dim, log_target_density, tuning_flag)
        final_emp_accept[i] = np.mean(output['accept_probs'])
    plt.scatter(np.log(dims), final_emp_accept)
    plt.show()


def test_pCN(phi):

    num_samples = 1000
    dim = 20
    s = 0.9

    output = pCN(num_samples, dim, phi, s)
    x = output['samples'][:, 0]
    y = output['samples'][:, 1]
    z = output['samples'][:, 2]


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(x, y, 'x')
    ax1.set_title('First and second coordinates')
    ax1.axis('equal')
    ax2.plot(output['accept_probs'], 'x')
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_title('Acceptance probabilities')
    ax3.plot(output['step_sizes'], 'x')
    ax3.set_title('Jump Sizes')
    ax4.plot(z, 'x')
    ax4.set_title('Third coordinate')
    plt.show()
    """
    if dim >= 4:
        df = pd.DataFrame(output['samples'])[[0,1,2,3]]
        g = sns.PairGrid(df, diag_sharey=False)
        g.map_upper(sns.scatterplot, s=15)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot, lw=2)
        plt.show()
    """
