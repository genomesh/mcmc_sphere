from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import numpy as np
import matplotlib.pyplot as plt
import emcee
import json


# Could save ESS and lag 1 autocorrelation in metadata, but haven't done yet.

def get_ess(seq, plot_bool = False):

    rho = np.corrcoef([seq[:-1], seq[1:]])[0,1]

    print(f'autocorrelation: {rho}\n')

    ess = (1 - rho) / (1 + rho) * len(seq)

    print(f'Effective Sample Size: {ess:.2f}\n')

    if plot_bool:

        plot_acf(seq)
        plot_pacf(seq)
        plt.show()

    return [rho, ess]


# old function
#def plot_iact(folders):

    fig, ax = plt.subplots()

    for i in range(len(folders)):

        folder = './output/' + folders[i]
        print(folder)

        samples = get_prob_samples(folder)

        get_ess(samples, plot_bool=False)[1]

        iact = emcee.autocorr.integrated_time(samples)

        print(iact)

        ax.plot([10,20,30,40,50][i], iact, 'r_', markersize = 4)

    ax.set(xlabel='dimension', ylabel='IACT',
        title= 'Integrated Autocorrelation time (IACT) vs dimension')
    ax.grid()

    #fig.savefig("u_plot.png")
    plt.show()

def update_correlation(folder, samples):

    rho, ess = get_ess(samples)
    iact = emcee.autocorr.integrated_time(samples)[0]

    with open(folder + '/metadata.json', 'r+') as f:
        data = json.load(f)
        data['ess'] = ess
        data['autocorrelation'] = rho
        data['iact'] = iact
        data['mc_estimate'] = np.mean(samples)

        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4)
        f.truncate()     # remove remaining part
    return True
