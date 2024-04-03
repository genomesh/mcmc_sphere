import numpy as np
import pandas as pd

coal_data = pd.read_csv("coal.csv")

coal_data['scaled_year'] = (coal_data['Year'] - 1851) / (1962 - 1851)

beta_samples = np.load('./output/mixed_beta_samples.npy')


def normalise(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.array([1] + [0] * (len(x) - 1))
    return x / norm

def coal_potential(x):
    if np.linalg.norm(x) != 1:
        x = normalise(x)
    outersum = 0
    dimension = len(x)
    for j, year in enumerate(coal_data['scaled_year']):
        if coal_data['Count'][j] > 0:
            innersum = 0
            for i in range(dimension):
                if i == 0:
                    phi = 1
                else:
                    phi = np.sqrt(2) * np.cos(np.pi * i * year)
                innersum += x[i] * phi
            outersum += coal_data['Count'][j] * np.log(np.abs(innersum))
    return -2 * outersum

def generate_beta_sphere_potential(dy):

    def beta_sphere_potential(x):

        if np.linalg.norm(x) != 1:
            x = normalise(x)

        outersum = 0
        dimension = len(x)
        for data_point in beta_samples[0:dy]:
            innersum = 0
            for i in range(dimension):
                if i == 0:
                    phi = 1
                else:
                    phi = np.sqrt(2) * np.cos(np.pi * i * data_point)
                innersum += x[i] * phi
            outersum += np.log(np.abs(innersum))
        return -2 * outersum
    
    return beta_sphere_potential
