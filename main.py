from algorithms import *
from visuals import *

#folder = rw_mcmc(5000, 4, 'phi2', 'id')
#folder = pCN(10000, 4, 'phi2', 'id')
#folder = rw_mcmc_tuning(5000, 4, 'phi2', 'id')

folder = rw_mcmc(5000, 4, 'beta_mix', 'inv_lap')

print(folder)

basic_plots(folder)

mcmc_plots(folder)
