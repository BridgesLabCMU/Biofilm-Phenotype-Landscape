import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.optim import Adam

import clip

from model.utils import device, local_rank, rank
from model.loss import SimCLR
from model.model import ViT

import os
import json
import re
import time


GPU_COUNT = 8
LEARNING_RATE = 0.003
EPSILON = 1e-4
N_EPOCHS = 100
BATCH_SIZE = 54
CRITERION = SimCLR()


with open("../../config/train_config.json", "r") as f:
    args = json.load(f)
f.close()

for param in [args['dataloader'], args['weights']]:
    if not param:
        continue
    regex_match = re.match(r".*\.pth", param)
    if not regex_match:
        print("ERROR: Input and output files must follow the format filename.pth")
        exit()

dataloader_filename = args['dataloader']
weights_filename = args['weights']


data_loc = args['data_loc']
dataloader_loc = args['dataloader_loc']
weights_loc = args['weights_loc']





def train_model(model, optimizer, dataloader, sampler, num_epochs, criterion):
    """
    Desc: Main training function, updates model parameters using self-supervised learning via SimCLR, loss function is infoNCE
    Input: Batch of augmented videos passed into separate GPUs, coordinated using sampler
    """
    model.train()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")
        running_loss = 0.0
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader):
            print(f"Batch {i + 1} / {len(dataloader)}")

            batch_start = time.time()
            optimizer.zero_grad()
                    
            augmented1, augmented2
                            
            augmented1 = augmented1.to(device)
            augmented2 = augmented2.to(device)
            
            embeddings1 = model(augmented1, "train")
            embeddings2 = model(augmented2, "train")
            
            loss = criterion(embeddings1, embeddings2, "infoNCE")
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f"Loss: {loss.item():.4f}, Time: {time.time() - batch_start} seconds")

    print(f"Epoch: {epoch + 1} / {num_epochs}, Loss: {running_loss:.4f}")



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
        
        print("GPU COUNT", GPU_COUNT)
        if GPU_COUNT > 1:
            print(f"Using {GPU_COUNT} GPUs")
            
            init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=GPU_COUNT,
                rank=rank
            ) 
            vit = DistributedDataParallel(vit, device_ids=[local_rank], output_device=local_rank).module
            sampler = DistributedSampler()
        
        
        print(f"Loading pre-trained VIT model took {time.time() - model_load_start} seconds")
        print()
        
        print(f"Finetuning model...")
        print("Initial mem allocated", torch.cuda.memory_allocated())
        finetune_start = time.time()
        optimizer = Adam(vit.parameters(), lr=LEARNING_RATE, eps=EPSILON)
        
        dataset = torch.load(f"{dataloader_loc}/{dataloader_filename}", weights_only=False)
        sampler = DistributedSampler(dataset, shuffle=True, num_replicas=GPU_COUNT)
        dataloader = DataLoader(dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=(sampler is None), 
                                sampler=sampler)
        
        
        train_model(vit, optimizer, dataloader, sampler=sampler, num_epochs=N_EPOCHS, criterion=CRITERION)
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
    
    


