import torch
from torch.utils.data import DataLoader
import clip

from model.utils import device
from model.model import ViT

import time
import os

import numpy as np

import os
import json
import re
import time




with open("../../config/eval_config.json", "r") as f:
    args = json.load(f)
f.close()

for param in [args['dataloader'], args['weights']]:
    if not param:
        continue
    regex_match = re.match(r".*\.pth", param)
    if not regex_match:
        print("ERROR: Input and output files must follow the format filename.pth")
        exit()

dataloader_filename = args["dataloader"]
weights_filename = args["weights"]


data_loc = args['data_loc']
dataloader_loc = args['dataloader_loc']
weights_loc = args['weights_loc']


def eval_model(model, dataloader):
    """
    Computes video embeddings.
    Arguments:
    model - weights and biases for pretrained model
    dataloader - pth file with cropped and normalized grayscale images
    num_frames - number of frames
    keep_strains - strains to keep in analysis
    """
    with torch.no_grad():
        embedding_dims = model.output[0].in_features
        embeddings = torch.empty((len(dataloader), embedding_dims), device=device)
        labels = []
        for i, data in enumerate(dataloader):
            print(f"{i}/{len(dataloader)}")
            
            
            video, label = data[0], data[1]
            embedding = model(video)
            embeddings[i] = embedding
            labels.append(label)
        embeddings = embeddings.detach().cpu.numpy()
        labels = np.array(labels)
    return embeddings, labels


if __name__ == "__main__":
    with torch.no_grad():
        dataset = torch.load(f"{dataloader_loc}/{dataloader_filename}")
        dataloader = DataLoader(dataset, shuffle=False)
        
        if weights_filename in os.listdir(weights_loc):
            print(f"Loading finetuned vision transformer weights...")
            model_load_start = time.time()
            
            model, _ = clip.load("ViT-B/32", device=device)
            vision_transformer = ViT(model).to(device)
            
            checkpoint = torch.load(f"{weights_loc}/{weights_filename}")
            vision_transformer.load_state_dict(checkpoint)
            vision_transformer = vision_transformer.to(device)
            print(f"Loading finetuned VIT model took {time.time() - model_load_start} seconds")
            print()
        else:
            
            print(f"Loading pretrained vision transformer weights...")
            model_load_start = time.time()
            model, _ = clip.load("ViT-B/32", device=device)
            vision_transformer = ViT(model).to(device)
            print(f"Loading pre-trained VIT model took {time.time() - model_load_start} seconds")
            print()
        
        print(f"Evaluating model...")
        eval_start = time.time()
        embeddings, labels= eval_model(vision_transformer, dataloader)
        print(f"Evaluating model took {time.time() - eval_start} seconds")
        
        
        print("Saving embeddings...")
        save_start = time.time()
        np.save("../processed_data/embeddings.npy", embeddings)
        np.save("../../processed_data/labels.npy", labels)
        print(f"Saving embeddings took {time.time() - save_start} seconds")
        print()
            
