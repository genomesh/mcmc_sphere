from pathlib import Path
import numpy as np
import json

def save_outputs(metadata, **kwargs):
    
    folder_name = metadata['algorithm'] + '_' + metadata['potential'] + '_' + metadata['prior_cov'] + '_dim' + str(metadata['dimension'])

    Path("./output/" + folder_name).mkdir(parents=True, exist_ok=True)
    i = 1
    while Path("./output/" + folder_name + '/run' + str(i)).exists() and i < 100:
        i += 1
    run_folder = "./output/" + folder_name + '/run' + str(i)
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    with open(run_folder + '/metadata.json', "w") as outfile: 
        json.dump(metadata, outfile)

    for key, value in kwargs.items():
        np.save(run_folder + '/' + key + '.npy', value)

    return run_folder
