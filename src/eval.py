import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, ToPILImage,  Normalize, InterpolationMode, RandomResizedCrop, GaussianBlur
import clip

from model.utils import device
from model.loss import SimCLR, MultiClassNPairLoss
from model.model import ViT

import time
import os

import pandas as pd
import numpy as np
import cv2
from PIL import Image


import os
import json
import re
import time

from natsort import natsorted


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

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_pixels):
    # rotation_angles = [0, 90, 180, 270]
    # random_int = torch.randint(0, 4, (1,))[0].item()
    # angle = rotation_angles[random_int]
    return Compose([
        Resize(n_pixels, interpolation = InterpolationMode.BICUBIC),
        # RandomResizedCrop(n_pixels, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_pixels),
        # GaussianBlur(3),
        _convert_image_to_rgb,
        # transforms.RandomRotation((angle, angle)),
        ToTensor(),
        
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        # ToPILImage()
    ])

def eval_model(model, home_dir, num_frames, keep_strains):
    """
    Computes video embeddings.
    Arguments:
    home_dir - directory location of images
    num_frames - number of frames
    keep_strains - strains to keep in analysis
    """
    
    
    trans = _transform(224)
    data_dict = {}



    paths = os.listdir(home_dir)
    folders = []
    for entry in paths: 
        directory = f"{home_dir}/{entry}"
        if os.path.isdir(directory):
            if "Drawer" in directory:
                folders.append(f"{entry}")

    images_dirs = []
    for folder in folders:
        for sub_folder in os.listdir(f"{home_dir}/{folder}"):
            if os.path.isdir(f"{home_dir}/{folder}/{sub_folder}/results_images"):
                images_dirs.append(f"{home_dir}/{folder}/{sub_folder}/results_images")

            

    
    
    labels_dict = {}
    labels = pd.read_csv("../data/ReplicatePositions.csv")
    for _, row in labels.iterrows():
        labels_dict[row.iloc[0]] = row.iloc[1]



    embedding_dim = model.output[0].in_features
    embeddings = np.empty((144 * len(keep_strains), embedding_dim))
    labels = np.array([])
    ids = np.array([])
    i = 0
    for plate, dir in enumerate(images_dirs):
        path = dir
        for file in natsorted(os.listdir(path)):
            if file.find("mask") == -1 and file.find("Thumb") == -1:
                magnification = ""
                if file.find("4x") > 0:
                    magnification = "4x"
                    continue
                elif file.find("10x") > 0:
                    magnification = "10x"
                elif file.find("20x") > 0:
                    magnification = "20x"
                    continue
                elif file.find("40x") > 0:
                    magnification = "40x"
                    continue
                
                
                well = file[:3]
                
                if well[-1] == '_':
                    well = well[:2]
                strain = labels_dict[well]
                if strain not in keep_strains:
                    continue
                print(i+1)
                
                embeddings_dir = f"{home_dir}/Embeddings/{strain}"
                print("MAGNIFICATION: ", magnification)
                if not os.path.exists(f"{embeddings_dir}/{magnification}"):
                    os.makedirs(f"{embeddings_dir}/{magnification}")
                embeddings_dir = f"{home_dir}/Embeddings/{strain}/{magnification}"

                image_stack = []
                file_path = f"{path}/{file}"
                print(f"Reading image stack for {file_path}, strain {strain}")
                _,images = cv2.imreadmulti(mats=image_stack,
                                             filename=file_path,
                                             start=0,
                                             count=num_frames,
                                             flags=cv2.IMREAD_ANYCOLOR)
                if strain not in data_dict.keys():
                    data_dict[strain] = []
                
                
                video = []
                for image in images:
                    pil_image = Image.fromarray(image)
                    tensor_image = trans(pil_image)
                    video.append(tensor_image)
                video = torch.stack(video)
                video = video.to(device=device)
                video_embedding = model(video, "eval")
                video_embedding = video_embedding.detach().cpu().numpy()
                embeddings[i] = video_embedding
                labels = np.append(labels, strain)
                ids = np.append(ids, f"{plate}_{well}")
                print()
                i += 1
    return embeddings, labels, ids


if __name__ == "__main__":
    keep_strains = ['WT', 'flaA', 'hapR', 'luxO_D47E', 'manA', 'potD1', 'rbmB', 'vpsL', 'vpvC_W240R']
    num_frames = 25
    if weights_filename in os.listdir(weights_loc):
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
        embeddings, labels, ids = eval_model(vision_transformer, f"{data_loc}", num_frames, keep_strains)
        print(f"Evaluating model took {time.time() - eval_start} seconds")
        
        
        print("Saving embeddings...")
        save_start = time.time()
        np.save("../processed_data/embeddings.npy", embeddings)
        np.save("../processed_data/labels.npy", labels)
        np.save("../processed_data/wells.npy", ids)
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
            embeddings, labels, ids = eval_model(vision_transformer, f"{data_loc}", num_frames, keep_strains)
            print(f"Evaluating model took {time.time() - eval_start} seconds")
            
            print("Saving embeddings...")
            save_start = time.time()
            np.save("../processed_data/embeddings.npy", embeddings)
            np.save("../processed_data/labels.npy", labels)
            np.save("../processed_data/wells.npy", ids)
            print(f"Saving embeddings took {time.time() - save_start} seconds")
            print()
                
