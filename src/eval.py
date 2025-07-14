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


with open("../config/config.json", "r") as f:
    args = json.load(f)
f.close()

for param in args["eval"]["dataloader"], args["eval"]["weights"]:
    if not param:
        continue
    regex_match = re.match(r".*\.pth", param)
    if not regex_match:
        print("ERROR: Input and output files must follow the format filename.pth")
        exit()

dataloader_filename = args["eval"]["dataloader"]
weights_filename = args["eval"]["weights"]
hdf5_filename = f"{dataloader_filename[:dataloader_filename.find('.pth')]}.hdf5"


data_loc = f"../{args['data_loc']}"
dataloader_loc = f"../{args['dataloader_loc']}"
weights_loc = f"../{args['weights_loc']}"

def eval_model(model, dataloader):
    """
    Computes video embeddings.
    Arguments:
    model - vision transformer
    dataloader - raw images pth file
    keep_strains - strains to keep in analysis
    """
    with torch.no_grad():
        embedding_dim = model.output[0].in_features
        embeddings = np.empty((len(dataloader), embedding_dim))
        labels = np.array([])
        for i, data in enumerate(dataloader):
            print(f"Video {i+1} / {len(dataloader)}")
            video, strain = data[0], data[1]
            video = video.to(device)
            video_embedding = model(video, "eval")
            video_embedding = video_embedding.detach().cpu().numpy()
            embeddings[i] = video_embedding
            labels = np.append(labels, strain)
    return embeddings, labels


if __name__ == "__main__":
    dataset = torch.load(f"{dataloader_loc}/{dataloader_filename}", weights_only=False)
    dataloader = DataLoader(dataset)
    if weights_filename in os.listdir(weights_loc):
        with torch.no_grad():
            print(f"Loading finetuned vision transformer weights...")
            model_load_start = time.time()
            
            model, _ = clip.load("ViT-B/32", device=device)
            vision_transformer = ViT(model).to(device)
            
            checkpoint = torch.load(f"{weights_loc}/{weights_filename}")
            vision_transformer.load_state_dict(checkpoint)

            print(f"Loading finetuned VIT model took {time.time() - model_load_start} seconds")
            print()
            
            print(f"Evaluating model...")
            eval_start = time.time()
            embeddings, labels = eval_model(vision_transformer, dataloader)
            print(f"Evaluating model took {time.time() - eval_start} seconds")
            
            
            print("Saving embeddings...")
            save_start = time.time()
            np.save("../processed_data/embeddings.npy", embeddings)
            np.save("../processed_data/labels.npy", labels)
            print(f"Saving embeddings took {time.time() - save_start} seconds")
            print()
    else:
        with torch.no_grad():
            print(f"Loading pretrained vision transformer weights...")
            model_load_start = time.time()
            model, _ = clip.load("ViT-B/32", device=device)
            vision_transformer = ViT(model).to(device)
            print(f"Loading pre-trained VIT model took {time.time() - model_load_start} seconds")
            print()
            
            print("Evaluating model...")
            eval_start = time.time()
            embeddings, labels= eval_model(vision_transformer, dataloader)
            print(f"Evaluating model took {time.time() - eval_start} seconds")
            
            print("Saving embeddings...")
            save_start = time.time()
            np.save("../processed_data/embeddings.npy", embeddings)
            np.save("../processed_data/labels.npy", labels)
            print(f"Saving embeddings took {time.time() - save_start} seconds")
            print()
                
