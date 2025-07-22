import dataloaders.train_dataloader as tdl
import dataloaders.eval_dataloader as edl
import torch

import os
import json
import re
import time

import json

with open("../config/config.json", "r") as f:
    args = json.load(f)
f.close()

NUM_FRAMES = 25

for param in [args["train"]["dataloader"], args["train"]["weights"]]:
    if not param:
        continue
    regex_match = re.match(r".*\.pth", param)
    if not regex_match:
        print("ERROR: Input and output files must follow the format filename.pth")
        exit()

train_dataloader_filename = args["train"]["dataloader"]
eval_dataloader_filename = args["eval"]["dataloader"]
weights_filename = args["train"]["weights"]


data_loc = f"../{args['data_loc']}"
dataloader_loc = f"../{args['dataloader_loc']}"
weights_loc = f"../{args['weights_loc']}"

if train_dataloader_filename not in os.listdir(dataloader_loc) and eval_dataloader_filename not in os.listdir(dataloader_loc):
    print(f"Loading data into {train_dataloader_filename} and {eval_dataloader_filename}...")
    dataloader_start = time.time()
    
    train_data_dict = tdl.build_dataloader(data_loc, NUM_FRAMES)
    train_dataset = tdl.DictionaryDataset(train_data_dict)
    
    eval_data_dict = edl.build_dataloader(data_loc, NUM_FRAMES)
    eval_dataset = tdl.DictionaryDataset(eval_data_dict)
    
    print(f"Loading data took {time.time() - dataloader_start} seconds")
    print()

    print("Saving dataloaders...")
    dataloader_save_start = time.time()
    torch.save(train_data_dict, f"{dataloader_loc}/{train_dataloader_filename}")
    torch.save(eval_data_dict, f"{dataloader_loc}/{eval_dataloader_filename}")
    print(f"Saving dataloaders took {time.time() - dataloader_save_start} seconds")
    print()