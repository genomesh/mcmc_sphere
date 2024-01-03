import numpy as np
from scipy.integrate import quad

beta_samples = np.load('./output/mixed_beta_samples.npy')

def u(x, xis):
    v = 0
    for i, xi in enumerate(xis):
        v += xi * np.sin(x * (i+1))
    return v

def integrand(x, *xis):
    return np.exp(u(x, xis))    

def beta_mixture_potential(xis):
    dy = 1000
    I = quad(integrand, 0, 1, args = tuple(xis))[0]
    #print(f'value of integral: {I}')
    pot = dy * np.log(I)
    for y in beta_samples[0:dy]:
        pot -= u(y, xis)
    return pot

#beta_mixture_potential(np.array([1, 1, 4]))