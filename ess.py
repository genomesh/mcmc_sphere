from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import numpy as np
from matplotlib import pyplot

# Could save ESS and lag 1 autocorrelation in metadata, but haven't done yet.

def get_ess(folder, plot_bool = False):

    samples = np.load(folder + '/samples.npy')

    # just considering first coordinate.
    series = samples[:,1]

    rho = np.corrcoef([series[:-1], series[1:]])[0,1]

    print(f'autocorrelation: {rho}\n')

    ess = (1 - rho) / (1 + rho) * len(samples)

    print(f'Effective Sample Size: {ess:.2f}\n')

    if plot_bool:

        plot_acf(series)
        plot_pacf(series)
        pyplot.show()

    return ess

test_folders = [
    './output/tuned_rw_phi2_id_dim3/run1',
    './output/rw_phi2_id_dim500/run2',
    './output/pCN_phi2_id_dim4/run1'
]

get_ess(test_folders[2])