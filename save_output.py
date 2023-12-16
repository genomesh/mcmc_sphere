from pathlib import Path
import numpy as np
import json

def save_outputs(metadata, **kwargs):
    
    folder_name = metadata['algorithm'] + '_' + metadata['potential'] + '_' + metadata['prior_cov'] + '_dim' + str(metadata['dimension'])

    Path("./output/" + folder_name).mkdir(parents=True, exist_ok=True)
    i = 1
    while (Path("./output/" + folder_name + '/run' + str(i)).exists()):
        i += 1
        if i > 100:
            print('i over 100')
            break
    run_folder = "./output/" + folder_name + '/run' + str(i)
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    with open(run_folder + '/metadata.json', "w") as outfile: 
        json.dump(metadata, outfile)

    for key, value in kwargs.items():
        np.save(run_folder + '/' + key + '.npy', value)

    return run_folder

"""
key_info = {
    'target': 'phi1',
    'algorithm': 'pCN',
    'dimension': 20
}

meta_data = {
    'runtime': 30.345,
    'number of samples': 1000,
    'ESS': 201,
    'empirical alpha': 0.23,
    'target': 'phi1',
    'algorithm': 'pCN',
    'step size': 0.5,
    'initial_cov': 'identity',
    'dimension': 20
}

samples = np.array([13.3453,24.2124,1.2442,0.1,3,17])

save_outputs(meta_data, samples)"""