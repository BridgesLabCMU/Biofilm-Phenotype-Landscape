import dataloaders.train_dataloader as tdl
import torch

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

aug_dataloader_filename = args["train"]["dataloader"]
raw_dataloader_filename = args["eval"]["dataloader"]
weights_filename = args["train"]["weights"]


data_loc = f"../{args["data_loc"]}"
dataloader_loc = f"../{args["dataloader_loc"]}"
weights_loc = f"../{args["weights_loc"]}"



num_frames = 31
keep_strains = ['WT', 'flaA', 'hapR', 'luxO_D47E', 'manA', 'potD1', 'rbmB', 'vpsL', 'vpvC_W240R']
classes = np.unique(keep_strains) # reorder classes

if raw_dataloader_filename not in os.listdir(dataloader_loc) and aug_dataloader_filename not in os.listdir(dataloader_loc):
    print(f"Loading data into {raw_dataloader_filename} and {aug_dataloader_filename}...")
if raw_dataloader_filename not in os.listdir(dataloader_loc) and aug_dataloader_filename not in os.listdir(dataloader_loc):
    print(f"Loading data into {raw_dataloader_filename} and {aug_dataloader_filename}...")
    dataloader_start = time.time()
    raw_data_dict, aug_data_dict = tdl.build_dataloader(data_loc, num_frames, keep_strains)
    
    raw_dataset = tdl.RawDictionaryDataset(raw_data_dict)
    aug_dataset = tdl.AugmentedDictionaryDataset(aug_data_dict)
    
    
    torch.save(raw_dataset, f"{dataloader_loc}/raw_dataset.pth")
    torch.save(aug_dataset, f"{dataloader_loc}/aug_dataset.pth")
    
