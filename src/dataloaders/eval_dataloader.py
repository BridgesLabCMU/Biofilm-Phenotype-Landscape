import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import time
import os
import h5py

import pandas as pd
import numpy as np

import cv2
import h5py

from natsort import natsorted



def eval_model(home_dir, num_frames, keep_strains):
    """
    Builds python dictionary of input data, keys represent strain names, values represent list of images in tensor form
    
    Arguments:
    home_dir - directory location of images
    """

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


    i = 0
    for dir in images_dirs:
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
                                             flags=cv2.IMREAD_GRAYSCALE)
                if strain not in data_dict.keys():
                    data_dict[strain] = []
                
                for image in images:
                    state = torch.get_rng_state()
                    tensor_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255
                    
                    
                    torch.set_rng_state(state)
                data_dict[strain].append([torch.stack(augmented_images1), torch.stack(augmented_images2)])
                print(i)
                print()
                i += 1
    return data_dict


