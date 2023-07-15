import h5py
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
import os

# roll39_h5_path = './roll/leadsheet_roll.h5'
# roll2_h5_path = './roll2/leadsheet_roll2.h5'
for file in os.listdir('./roll'):
    h5_file = os.path.basename(file)
    if '.ipynb_checkpoints' not in file and 'leadsheet_roll' not in file:
        roll39_h5_path = os.path.join('./roll/', h5_file)
        roll2_h5_path = os.path.join('./roll2', h5_file[:-3]+'2.h5')
        with h5py.File(roll39_h5_path, 'r') as h5roll:
            with h5py.File(roll2_h5_path, "w") as hf:
                name_list = list(h5roll.keys())

                for i in name_list:
                    pitched = h5roll[i][()][:38]
                    drums = h5roll[i][()][38]

                    placeholder = np.zeros((2, *pitched.shape[1:])).astype('bool')
                    placeholder[0] = np.any(pitched, axis=0)
                    placeholder[1] = drums

                    hf.create_dataset(i, data=placeholder) 
