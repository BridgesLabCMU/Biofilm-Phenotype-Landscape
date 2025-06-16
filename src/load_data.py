import dataloaders.train_dataloader as tdl

import torch
from torch.utils.data import DataLoader

import os
import json
import re
import time

import json

import numpy as np

with open("../config/config.json", "r") as f:
    args = json.load(f)
f.close()


for param in [args["train"]["dataloader"], args["train"]["weights"]]:
    if not param:
        continue
    regex_match = re.match(r".*\.pth", param)
    if not regex_match:
        print("ERROR: Input and output files must follow the format filename.pth")
        exit()

dataloader_filename = args["train"]["dataloader"]
weights_filename = args["train"]["weights"]


data_loc = f"../{args["data_loc"]}"
dataloader_loc = f"../{args["dataloader_loc"]}"
weights_loc = f"../{args["weights_loc"]}"


hdf5_filename = f"{dataloader_filename[:dataloader_filename.find('.pth')]}.hdf5"


num_frames = 25
keep_strains = ['WT', 'flaA', 'hapR', 'luxO_D47E', 'manA', 'potD1', 'rbmB', 'vpsL', 'vpvC_W240R']
classes = np.unique(keep_strains) # reorder classes

if dataloader_filename in os.listdir(dataloader_loc):
    if hdf5_filename not in os.listdir(dataloader_loc):
        print(f"Loading data from {dataloader_filename}...")
        dataloader_start = time.time()
        dataloader = torch.load(f"{dataloader_loc}/{dataloader_filename}", weights_only=False)
        print(f"Loading data took {time.time() - dataloader_start} seconds")
        print()

        # print("Writing file to hdf5...")
        # convert_to_hdf5_time = time.time()
        # tdl.save_to_hdf5(dataloader, f"{dataloader_loc}/{hdf5_filename}")
        # print(f"Converting .pth file to .hdf5 took {time.time() - convert_to_hdf5_time} seconds")

        # print()
        

elif dataloader_filename not in os.listdir(dataloader_loc):
    print(f"Loading data into {dataloader_filename}...")
    dataloader_start = time.time()
    data_dict = tdl.build_dataloader(data_loc, num_frames, keep_strains)
    dataset = tdl.DictionaryDataset(data_dict)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)


    print(f"Loading data took {time.time() - dataloader_start} seconds")
    print()

    print("Saving dataloader...")
    dataloader_save_start = time.time()
    torch.save(dataloader, f"{dataloader_loc}/{dataloader_filename}")
    print(f"Saving dataloader took {time.time() - dataloader_save_start} seconds")
    print()

    # print("Writing file to hdf5...")
    # print(f"Loading data from {dataloader_filename}...")
    # dataloader_start = time.time()
    # dataloader = torch.load(f"{dataloader_loc}/{dataloader_filename}", weights_only=False)
    # print(f"Loading data took {time.time() - dataloader_start} seconds")
    # print()

    
    # convert_to_hdf5_time = time.time()
    # tdl.save_to_hdf5(dataloader, f"{dataloader_loc}/{hdf5_filename}")
    # print(f"Converting .pth file to .hdf5 took {time.time() - convert_to_hdf5_time} seconds")

    # print()