import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor,  Normalize, InterpolationMode, GaussianBlur, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

import os


from PIL import Image
import pandas as pd
import numpy as np
import re
import cv2

from natsort import natsorted

class RawDictionaryDataset(Dataset):
    """
    Description: Loads Dataset object from dictionary, dictionary contains number of values equal to size of dataset.
                 Generates 2 lists, one with associated class for each sample in dataset, one with associated videos with resized frames.
    Arguments:
    data_dict - keys: strain names, values: list of videos associated with each strain
    """
    def __init__(self, data_dict):
        self.videos = []
        self.strains = []
        for strain, videos in data_dict.items():
            for video in videos:
                self.videos.append(video)
                self.strains.append(strain)
        self.videos = torch.tensor(np.stack(self.videos), dtype=torch.float32)
            
    def __getitem__(self, index):
        return self.videos[index], self.strains[index]
    
    def __len__(self):
        return len(self.strains)
    

def raw_transform(n_pixels):
    """
    Description: OpenAI raw image transform, resizes image down to 224
    """
    return Compose([
        Resize(n_pixels, interpolation = InterpolationMode.BICUBIC),
        CenterCrop(n_pixels),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def build_dataloader(home_dir, num_frames, mutants_or_transposons, keep_strains):
    """
    Description: Builds python dictionary of input data, keys represent strain names, values represent list of images in tensor form
    Arguments:
    home_dir - directory location of images
    num_frames - number of frames in video
    mutants_or_transposons - boolean value for whether we read in mutant images or transposon images, mainly for handling directory structure of either set of images
    """
    raw_data_dict = {}

    paths = os.listdir(home_dir)
    folders = []
    for entry in paths: 
        directory = f"{home_dir}/{entry}"
        if os.path.isdir(directory):
            if "Drawer" in directory:
                folders.append(f"{directory}")
    if mutants_or_transposons == "mutants":
        images_dirs = []
        for folder in natsorted(folders):
            for sub_folder in [f.path for f in os.scandir(folder) if f.is_dir()]:
                if "Plate" in sub_folder:
                    images_dirs.append(f"{sub_folder}/results_images")
    elif mutants_or_transposons == "transposons":
        images_dirs = []
        for folder in natsorted(folders):
            for sub_folder in [f.path for f in os.scandir(folder) if f.is_dir()]:
                if "Processed images" in sub_folder:
                    images_dirs.append(f"{sub_folder}")
        
    if mutants_or_transposons == "mutants": 
        labels_dict = {}
        labels = pd.read_csv(f"{home_dir}/ReplicatePositions.csv")
        for _, row in labels.iterrows():
            labels_dict[row.iloc[0]] = row.iloc[1]


    i = 0
    for dir in images_dirs:
        path = dir
        for file in natsorted(os.listdir(path)):
            if file.find("mask") == -1 and file.find("Thumb") == -1:
                if mutants_or_transposons == "mutants":
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
                elif mutants_or_transposons == "transposons":
                    well_id = file[:file.find("_")]
                    plate_well = f"{i+1}-{well_id}"
                print(i+1)
                
                image_stack = []
                file_path = f"{path}/{file}"
                print(f"Reading image stack for {file_path}")
                _,images = cv2.imreadmulti(mats=image_stack,
                                             filename=file_path,
                                             start=0,
                                             count=num_frames,
                                             flags=cv2.IMREAD_ANYCOLOR)
                if mutants_or_transposons == "mutants":
                    if strain not in raw_data_dict.keys():
                        raw_data_dict[strain] = []
                elif mutants_or_transposons == "transposons":
                    if plate_well not in raw_data_dict.keys():
                        raw_data_dict[plate_well] = []
                
                raw_images = []
                
                raw_trans = raw_transform(224)
                
                for image in images:
                    pil_image = Image.fromarray(image)
                    raw_image = raw_trans(pil_image)
                    raw_images.append(raw_image)
                    
                if mutants_or_transposons == "mutants":
                    raw_data_dict[strain].append(np.stack(raw_images))
                elif mutants_or_transposons == "transposons":
                    raw_data_dict[plate_well].append(np.stack(raw_images))
                print()
                i += 1
    return raw_data_dict

