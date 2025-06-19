

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_32, ViT_B_32_Weights
import clip

from model.utils import device
from model.loss import SimCLR, MultiClassNPairLoss, SupCon
from model.model import ViT

import numpy as np
import pandas as pd

import os
import json
import h5py
import re
import time

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
hdf5_filename = f"{dataloader_filename[:dataloader_filename.find('.pth')]}.hdf5"


data_loc = f"../{args["data_loc"]}"
dataloader_loc = f"../{args["dataloader_loc"]}"
weights_loc = f"../{args["weights_loc"]}"





def train_model(model, optimizer, criterion, dataloader, batch_size, num_epochs):
    """
    Desc: Main training function, uses model paramete, optimizer, loss function, dataloader, batch_size, and number of epochs
    Takes batch_size samples of augmented videos 
    """
    embedding_dims = model.model.ln_final.normalized_shape[0]
    model.train()
    
    with h5py.File(dataloader, 'r') as f:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1} / {num_epochs}")
            running_loss = 0.0
            
            for i, batch in enumerate(f.keys()):
                # print(round(torch.cuda.memory_allocated() / 1000000000, 2))
                print(f"Batch {i + 1} / {len(f.keys())}")

                batch_start = time.time()
                optimizer.zero_grad()
                
                videos = torch.tensor(f[batch]["videos"][()], device=device)
                strains = torch.tensor(f[batch]["strains"][()], device=device)
                
                
                
                augmented1 = videos[0]
                augmented2 = videos[1]
                    
                    
                
                
            # for i, batch in enumerate(dataloader):
            #     print(f"Batch {i+1} / {len(dataloader)}")
                
            #     embeddings1 = torch.empty(batch_size, embedding_dims)
            #     embeddings2 = torch.empty(batch_size, embedding_dims)
            #     batch_start = time.time()
            #     optimizer.zero_grad()
                

                
                # augmented1, augmented2, strains = batch[0], batch[1], batch[2]
                                
                # augmented1 = augmented1.to(device)
                # augmented2 = augmented2.to(device)
                
                embeddings1 = model(augmented1, "train")
                embeddings2 = model(augmented2, "train")

                
                
                # embeddings = torch.cat((embeddings1, embeddings2), dim=0)
                # loss = criterion(embeddings, strains)
                similarity, loss = criterion(embeddings1, embeddings2)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # print(pd.DataFrame(similarity))
                print(f"Loss: {loss.item():.4f}, Time: {time.time() - batch_start} seconds")

        print(f"Epoch: {epoch + 1} / {num_epochs}, Loss: {running_loss:.4f}")





num_frames = 25
num_epochs = 100
keep_strains = ['WT', 'flaA', 'hapR', 'luxO_D47E', 'manA', 'potD1', 'rbmB', 'vpsL', 'vpvC_W240R']
classes = np.unique(keep_strains) # reorder classes
batch_size = 144


if weights_filename not in os.listdir(weights_loc):
    print(f"Loading pretrained weights...")
    model_load_start = time.time()
    model, _ = clip.load("ViT-B/32", device=device)
    for param in model.transformer.resblocks[:-3].parameters():
        param.requires_grad = False
    for param in model.transformer.resblocks[-3:].parameters():
        param.requires_grad = True
        
    vit = ViT(model).to(device)
    print(f"Loading pre-trained VIT model took {time.time() - model_load_start} seconds")
    print()
    
    print(f"Finetuning model...")
    print("Initial mem allocated", torch.cuda.memory_allocated())
    finetune_start = time.time()
    optimizer = torch.optim.Adam(vit.parameters(), lr=0.0000001, eps=1e-4)
    criterion = SimCLR()

    # dataloader = torch.load(f"{dataloader_loc}/{dataloader_filename}", weights_only=False)
    
    train_model(vit, optimizer, criterion, f"{dataloader_loc}/{hdf5_filename}", batch_size=batch_size, num_epochs=num_epochs)
    print(f"Finetuning model took {time.time() - finetune_start} seconds")
    print()
    
    print("Saving finetuned model weights...")
    weights_save_start = time.time()
    torch.save(vit.state_dict(), f"{weights_loc}/{weights_filename}")
    print(f"Saving model weights took {time.time() - weights_save_start} seconds")
    print()
    
    


