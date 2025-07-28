import dataloaders.train_dataloader as tdl
import dataloaders.eval_dataloader as edl
import torch

import os
import json
import re
import time

import json

with open("../../config/config.json", "r") as f:
    args = json.load(f)
f.close()

NUM_FRAMES = 25

for param in [args["dataloader"], args["weights"]]:
    if not param:
        continue
    regex_match = re.match(r".*\.pth", param)
    if not regex_match:
        print("ERROR: Input and output files must follow the format filename.pth")
        exit()

train_dataloader_filename = args["dataloader"]
eval_dataloader_filename = args["dataloader"]
weights_filename = args["weights"]
mutants_or_transposons = args["mutants_or_transposons"]

data_loc = args['data_loc']
dataloader_loc = args['dataloader_loc']
weights_loc = args['weights_loc']
keep_strains = ['WT', 'flaA', 'hapR', 'luxO_D47E', 'potD1', 'rbmB', 'vpsL', 'vpvC_W240R']
if train_dataloader_filename not in os.listdir(dataloader_loc) and eval_dataloader_filename not in os.listdir(dataloader_loc):
    print(f"Loading data into {train_dataloader_filename} and {eval_dataloader_filename}...")
    dataloader_start = time.time()
    
    train_data_dict = tdl.build_dataloader(data_loc, NUM_FRAMES, mutants_or_transposons, keep_strains)
    train_dataset = tdl.AugmentedDictionaryDataset(train_data_dict)
    
    eval_data_dict = edl.build_dataloader(data_loc, NUM_FRAMES, mutants_or_transposons, keep_strains)
    eval_dataset = edl.RawDictionaryDataset(eval_data_dict)
    
    print(f"Loading data took {time.time() - dataloader_start} seconds")
    print()

    print("Saving dataloaders...")
    dataloader_save_start = time.time()
    torch.save(train_dataset, f"{dataloader_loc}/{train_dataloader_filename}")
    torch.save(eval_dataset, f"{dataloader_loc}/{eval_dataloader_filename}")
    print(f"Saving dataloaders took {time.time() - dataloader_save_start} seconds")
    print()
