import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from pathlib import Path

# roll39_h5_path = './roll/leadsheet_roll.h5'
# roll2_h5_path = './roll2/leadsheet_roll2.h5'
input_path = '/opt/tiger/kinwai/jointist/MSD/roll/MSD_test_part01.h5' 
output_path = './MSD/sparse_roll/'
Path(output_path).mkdir(parents=True, exist_ok=True)
# for file in os.listdir('./MSD/roll/MTAT_roll39_full'):

# file = 'MSD_test_part02'
       
with h5py.File(input_path, 'r') as h5roll:
    name_list = list(h5roll.keys())
    for i in name_list:
        sparse_roll  = torch.tensor(h5roll[i][()]).to_sparse()
        torch.save(sparse_roll, os.path.join(output_path, f"{i}.pt"))
#                     sparse_roll = sparse.COO(h5roll[i][()])
#                     hf.create_dataset(i, data=sparse_roll) 
