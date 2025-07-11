import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, ToPILImage,  Normalize, InterpolationMode, RandomResizedCrop, GaussianBlur, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip


import time
import os
import h5py


from PIL import Image
import pandas as pd
import numpy as np

import cv2
import h5py

from natsort import natsorted

class RawDictionaryDataset(Dataset):
    def __init__(self, data_dict):
        self.videos = []
        self.strains = []
        for strain, videos in data_dict.items():
            for video in videos:
                self.videos.append(video)
                self.strains.append(strain)
        
        

        
        self.videos = torch.tensor(np.stack(self.videos), dtype=torch.float32)
        self.videos = self.videos.expand(-1, -1, 3, -1, -1)
        self.videos = torch.tensor(np.stack(self.videos))
        
        
        self.strain_names, self.strains_numeric = np.unique(self.strains, return_inverse=True)
        
        self.strains_numeric = torch.tensor(self.strains_numeric)

        self.strains = self.strains_numeric
            
    def __getitem__(self, index):
        return self.videos[index], self.strains[index]
    
    def __len__(self):
        return len(self.strains)

class AugmentedDictionaryDataset(Dataset):
    """
    Loads Dataset object from dictionary, dictionary contains number of values equal to size of dataset.
    Generates 2 lists, one with associated class for each sample in dataset, one with associated videos with resized frames.
    
    Arguments:
    data_dict - keys: strain names, values: list of videos associated with each strain
    """

    def __init__(self, data_dict):
        self.augmented_videos1 = []
        self.augmented_videos2 = []
        self.strains = []
        for strain, video_pair in data_dict.items():
            for augmented_video1, augmented_video2 in video_pair:
                self.augmented_videos1.append(augmented_video1)
                self.augmented_videos2.append(augmented_video2)
                self.strains.append(strain)
        
        self.augmented_videos1 = torch.tensor(np.stack(self.augmented_videos1), dtype=torch.float32)
        self.augmented_videos1 = self.augmented_videos1.expand(-1, -1, 3, -1, -1)
        self.augmented_videos1 = torch.tensor(np.stack(self.augmented_videos1))

        self.augmented_videos2 = torch.tensor(np.stack(self.augmented_videos2), dtype=torch.float32)
        self.augmented_videos2 = self.augmented_videos2.squeeze(2)
        self.augmented_videos2 = self.augmented_videos2.expand(-1, -1, 3, -1, -1)
        self.augmented_videos2 = torch.tensor(np.stack(self.augmented_videos2))



        # reorganize data so that each batch contains all classes
        num_classes = len(data_dict.keys())
        augmented_videos1_copy = torch.empty(self.augmented_videos1.shape)
        augmented_videos2_copy = torch.empty(self.augmented_videos2.shape)
        strains_copy = ["" for _ in range(len(self.strains))]
        step = int(len(self.strains) / num_classes)
        for i in range(0, len(self.strains), num_classes):
            augmented_videos1_copy[i:i + num_classes] = self.augmented_videos1[0::step]
            augmented_videos2_copy[i:i + num_classes] = self.augmented_videos2[0::step]
            strains_copy[i:i + num_classes] = self.strains[0::step]
        self.augmented_videos1 = augmented_videos1_copy
        self.augmented_videos2 = augmented_videos2_copy
        self.strains = strains_copy
        
        self.strain_names, self.strains_numeric = np.unique(self.strains, return_inverse=True)
        
        self.strains_numeric = torch.tensor(self.strains_numeric)

        self.strains = self.strains_numeric

    def __getitem__(self, index):
        return self.augmented_videos1[index], self.augmented_videos2[index], self.strains[index]
    
    def __len__(self):
        return len(self.strains)


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def raw_transform(n_pixels):
    return Compose([
        Resize(n_pixels, interpolation = InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def aug_transform(n_pixels):
    rotation_angles = [0, 90, 180, 270]
    random_int = torch.randint(0, 4, (1,))[0].item()
    angle = rotation_angles[random_int]
    return Compose([
        Resize(n_pixels, interpolation = InterpolationMode.BICUBIC),
        CenterCrop(n_pixels),
        RandomRotation((angle, angle)),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        GaussianBlur(3),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def build_dataloader(home_dir, num_frames, keep_strains):
    """
    Builds python dictionary of input data, keys represent strain names, values represent list of images in tensor form
    
    Arguments:
    home_dir - directory location of images
    """
    
   

    raw_data_dict = {}
    aug_data_dict = {}

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
                if strain not in aug_data_dict.keys():
                    aug_data_dict[strain] = []
                if strain not in raw_data_dict.keys():
                    raw_data_dict[strain] = []

                
                raw_images = []
                augmented_images1 = []
                augmented_images2 = []
                
                raw_trans = raw_transform(224)
                aug_trans1 = aug_transform(224)
                aug_trans2 = aug_transform(224)
                
                for image in images:
                    pil_image = Image.fromarray(image)
                    
                    state = torch.get_rng_state()
                    
                    
                    raw_image = raw_trans(pil_image)
                    raw_images.append(raw_image)
                    
                    augmented_image1 = aug_trans1(pil_image)
                    augmented_images1.append(augmented_image1)
                    
                    augmented_image2 = aug_trans2(pil_image)
                    augmented_images2.append(augmented_image2)
                    
                    torch.set_rng_state(state)
                    

                # for k, (frame1, frame2) in enumerate(zip(augmented_images1, augmented_images2)):
                        
                #     image1 = transforms.functional.to_pil_image(frame1, mode="RGB")
                #     image2 = transforms.functional.to_pil_image(frame2, mode="RGB")
                    
                #     image1.save(f"../videos/augmented1/frame{k}.png", format="PNG")
                #     image2.save(f"../videos/augmented2/frame{k}.png", format="PNG")
                # exit()
                
                raw_data_dict[strain].append(np.stack(raw_images))
                aug_data_dict[strain].append([np.stack(augmented_images1), np.stack(augmented_images2)])
                
                print()
                i += 1
    return raw_data_dict, aug_data_dict

