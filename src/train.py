import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.optim import Adam

import torchvision.transforms as transforms
from torchvision.models import vit_b_32, ViT_B_32_Weights
import clip

from model.utils import device, local_rank, rank
from model.loss import SimCLR, MultiClassNPairLoss, SupCon
from model.model import ViT

import numpy as np
import pandas as pd

import argparse
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


data_loc = f"../{args['data_loc']}"
dataloader_loc = f"../{args['dataloader_loc']}"
weights_loc = f"../{args['weights_loc']}"





def train_model(model, optimizer, criterion, dataloader, num_epochs, sampler):
    """
    Desc: Main training function, updates model parameters using self-supervised learning via SimCLR, loss function is infoNCE
    Input: Batch of augmented videos passed into separate GPUs, coordinated using sampler
    """
    embedding_dims = model.model.ln_final.normalized_shape[0]
    model.train()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")
        running_loss = 0.0
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader):
            # print(round(torch.cuda.memory_allocated() / 1000000000, 2))
            print(f"Batch {i + 1} / {len(dataloader)}")

            batch_start = time.time()
            optimizer.zero_grad()
            
            # videos = torch.tensor(f[batch]["videos"][()], device=device)
            # strains = torch.tensor(f[batch]["strains"][()], device=device)
                    
            augmented1, augmented2, strains = data[0], data[1], data[2]
            # augmented1 = videos[0]
            # augmented2 = videos[1]
            
                
            
            
        # for i, batch in enumerate(dataloader):
        #     print(f"Batch {i+1} / {len(dataloader)}")
            
        #     embeddings1 = torch.empty(batch_size, embedding_dims)
        #     embeddings2 = torch.empty(batch_size, embedding_dims)
        #     batch_start = time.time()
        #     optimizer.zero_grad()
            

            
            # augmented1, augmented2, strains = batch[0], batch[1], batch[2]
                            
            augmented1 = augmented1.to(device)
            augmented2 = augmented2.to(device)
            
            embeddings1 = model(augmented1, "train")
            embeddings2 = model(augmented2, "train")

            
            
            # embeddings = torch.cat((embeddings1, embeddings2), dim=0)
            # loss = criterion(embeddings, strains)
            similarity, loss = criterion(embeddings1, embeddings2, "infoNCE")
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
batch_size = 54

# logging
@record
def main():
    if weights_filename not in os.listdir(weights_loc):
        print(f"Loading pretrained weights...")
        model_load_start = time.time()
        model, _ = clip.load("ViT-B/32")
        
        # freeze last 3 layers
        for param in model.transformer.resblocks[:-3].parameters():
            param.requires_grad = False
        for param in model.transformer.resblocks[-3:].parameters():
            param.requires_grad = True
            
        vit = ViT(model)
        vit.to(device)
        
        gpu_count = int(os.environ['CUDA_VISIBLE_DEVICES'])
        print("GPU COUNT", gpu_count)
        if gpu_count > 1:
            print(f"Using {gpu_count} GPUs")
            
            init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=gpu_count,
                rank=rank
            ) 
            vit = DistributedDataParallel(vit, device_ids=[local_rank], output_device=local_rank).module
            
        
        
        print(f"Loading pre-trained VIT model took {time.time() - model_load_start} seconds")
        print()
        
        print(f"Finetuning model...")
        print("Initial mem allocated", torch.cuda.memory_allocated())
        finetune_start = time.time()
        optimizer = Adam(vit.parameters(), lr=0.0003, eps=1e-6)
        criterion = SimCLR()


        dataset = torch.load(f"{dataloader_loc}/{dataloader_filename}", weights_only=False)
        sampler = DistributedSampler(dataset, num_replicas=8, shuffle=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=(sampler is None), 
                                sampler=sampler)
        
        
        train_model(vit, optimizer, criterion, dataloader, num_epochs=num_epochs, sampler=sampler)
        print(f"Finetuning model took {time.time() - finetune_start} seconds")
        print()
        
        print("Saving finetuned model weights...")
        weights_save_start = time.time()
        torch.distributed.barrier()
        
        # save model on first gpu
        if rank == 0:
            torch.save(vit.state_dict(), f"{weights_loc}/{weights_filename}")
            
        print(f"Saving model weights took {time.time() - weights_save_start} seconds")
        print()

if __name__ == "__main__":
    main()
    
    


