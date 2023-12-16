from algorithms import *
from visuals import *

folder = rw_mcmc(5000, 4, 'phi1', 'id')

print(folder)

#basic_plots(folder)

mcmc_plots(folder)