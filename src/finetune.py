import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_32, ViT_B_32_Weights
import torchvision.transforms as transforms

import time
import os
import argparse
import re

import pandas as pd
import numpy as np

from PIL import Image
import cv2
import h5py
import json

from natsort import natsorted

import matplotlib.pyplot as plt

from src.model.model import ViT, MultiClassNPairLoss, SimCLR, device



with open("config.json", "r") as f:
    args = json.load(f)
f.close()

# yes_args = ["Y", "y", "Yes", "yes"]
# no_args = ["N", "n", "No", "no"]

# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--dataloader_exists", required=True)
# parser.add_argument("-w", "--weights_exists", required=True)

# parser.add_argument("--dataloader_input", "--dataloader_input")
# parser.add_argument("--weights_input", "--weights_input")

# parser.add_argument("--dataloader_output", "--dataloader_output")
# parser.add_argument("--weights_output", "--weights_output")
# args = parser.parse_args()

# if args["dataloader_exists"] in yes_args and not args["dataloader"]:
#     print("""ERROR: Must provide dataloader input filename using --dataloader_input filename.pth
#         If no dataloader has been generated, rerun with -d No|no|N|n --dataloader_output filename.pth""")    
#     exit()

# if args["weights_exists"] in yes_args and not args["weights"]:
#     print("""ERROR: Must provide weights input filename using --weights_input filename.pth
#         If no model weights have been generated, rerun with -w No|no|N|n --weights_output filename.pth""")
#     exit()

# if args["dataloader_exists"] in no_args and not args["dataloader"]:
#     print("""ERROR: Must provide dataloader output filename using --dataloader_output filename.pth
#         If dataloader has been generated, rerun with -d Yes|yes|Y|y --dataloader_input filename.pth""")      
#     exit()

# if args["weights_exists"] in no_args and not args["weights"]:
#     print("""ERROR: Must provide weights output filename using --dataloader_output filename.pth
#         If model weights have been generated, rerun with -w Yes|yes|Y|y --weights_input filename.pth""")   
#     exit()   

for param in [args["dataloader"], args["dataloader"], args["weights"], args["weights"]]:
    if not param:
        continue
    regex_match = re.match(r".*\.pth", param)
    if not regex_match:
        print("ERROR: Input and output files must follow the format filename.pth")
        exit()



class DictionaryDataset(Dataset):
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
        num_classes = len(data_dict.keys())
        for strain, video_pair in data_dict.items():
            for augmented_video1, augmented_video2 in video_pair:
                self.augmented_videos1.append(augmented_video1)
                self.augmented_videos2.append(augmented_video2)
                self.strains.append(strain)
        
        self.augmented_videos1 = torch.tensor(np.stack(self.augmented_videos1), dtype=torch.float32)
        self.augmented_videos1 = self.augmented_videos1.squeeze(2)
        self.augmented_videos1 = self.augmented_videos1.expand(-1, -1, 3, -1, -1)
        self.augmented_videos1 = torch.tensor(np.stack(self.augmented_videos1))

        self.augmented_videos2 = torch.tensor(np.stack(self.augmented_videos2), dtype=torch.float32)
        self.augmented_videos2 = self.augmented_videos2.squeeze(2)
        self.augmented_videos2 = self.augmented_videos2.expand(-1, -1, 3, -1, -1)
        self.augmented_videos2 = torch.tensor(np.stack(self.augmented_videos2))


        augmented_videos1_copy = torch.empty(self.augmented_videos1.shape)
        augmented_videos2_copy = torch.empty(self.augmented_videos2.shape)
        strains_copy = ["" for _ in range(len(self.strains))]
        

        step = int(len(self.strains) / num_classes)


        # reorganize data so that each batch contains all classes
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




def build_data_dict(home_dir, num_frames, keep_strains):
    """
    Builds python dictionary of input data, keys represent strain names, values represent list of images in tensor form
    
    Arguments:
    home_dir - directory location of images
    """
    
    trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.GaussianBlur(3)]
            )

    data_dict = {}

    paths = os.listdir(home_dir)
    folders = []
    for entry in paths: 
        directory = f"./{entry}"
        if os.path.isdir(directory):
            if "Drawer" in directory:
                folders.append(f"{entry}")

    images_dirs = []
    for folder in folders:
        for sub_folder in os.listdir(folder):
            if os.path.isdir(f"./{folder}/{sub_folder}/results_images"):
                images_dirs.append(f"./{folder}/{sub_folder}/results_images")

    labels_dict = {}
    labels = pd.read_csv("ReplicatePositions.csv")
    for _, row in labels.iterrows():
        labels_dict[row.iloc[0]] = row.iloc[1]
    print(labels_dict)


    classes = np.unique(labels.iloc[:,1])
    print(home_dir)
    os.makedirs(f"{home_dir}/Embeddings", exist_ok=True)

    for c in classes:
        os.makedirs(f"{home_dir}/Embeddings/{c}", exist_ok=True)

    

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
                resized_images = []
                augmented_images1 = []
                augmented_images2 = []
                for image in images:
                    state = torch.get_rng_state()
                    tensor_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255

                    
                    rotation_angles = [0, 90, 180, 270]
                    random_int = torch.randint(0, 4, (1,))[0].item()
                    angle = rotation_angles[random_int]
                    
                    rotation_transform = transforms.RandomRotation((angle, angle))
                    

                    augmented_image1 = trans(tensor_image)
                    augmented_image1 = rotation_transform(augmented_image1)
                    augmented_images1.append(augmented_image1)
                    
                    augmented_image2 = trans(tensor_image)
                    augmented_image2 = rotation_transform(augmented_image2)
                    augmented_images2.append(augmented_image2)
                    
                    torch.set_rng_state(state)
                data_dict[strain].append([torch.stack(augmented_images1), torch.stack(augmented_images2)])
                print(i)
                print()
                i += 1
        #         if i == 4 * len(keep_strains):
        #             break_out = True
        #             break
        # if break_out:
        #     break
    return data_dict


def eval_model(model, hdf5_filename, classes):
    embedding_dims = model.model.heads[0].out_features
    embeddings = np.empty(embedding_dims)
    labels = np.array([])
    with h5py.File(hdf5_filename) as f:
        model.eval()
        
        for i, batch in enumerate(f.keys()):
            
            videos = torch.tensor(f[batch]["videos"][()], device=device)
            strains = torch.tensor(f[batch]["strains"][()], device=device)
            
            for j, (video, strain) in enumerate(zip(videos, strains)):
                print(f"Generating embeddings for batch {i + 1} / {len(f)}, strain {classes[strain]}")
                embedding = model(video).detach().numpy()

                embeddings = np.vstack((embeddings, embedding))
                labels = np.append(labels, classes[strain])
            print()
        f.close()
    
    return embeddings[1:], labels
            
def save_dataloader_to_hdf5(dataloader, hdf5_filename):
    with h5py.File(hdf5_filename, 'w') as f:
        # Iterate over the DataLoader
        for i, data in enumerate(dataloader):
            print(f"Batch {i + 1} / {len(dataloader)}")
            videos1, videos2, strains = data[0], data[1], data[2]
            
            # create group for current batch
            batch_group = f.create_group(f'batch_{i}')
            
            # tensor to np array

            videos_1_np = videos1.cpu().numpy()
            videos_2_np = videos2.cpu().numpy()

            videos_np = np.array([videos_1_np, videos_2_np])
            strain_np = strains.cpu().numpy()

            # videos dataset inside batch group
            batch_group.create_dataset('videos', data=videos_np)
            # strains dataset inside batch group
            batch_group.create_dataset('strains', data=strain_np)
        f.close()
    


home_dir = args["data_dir_name"]



num_frames = 25
num_epochs = 20
keep_strains = ['WT', 'flaA', 'hapR', 'luxO_D47E', 'manA', 'potD1', 'rbmB', 'vpsL', 'vpvC_W240R']
classes = np.unique(keep_strains) # reorder classes
embedding_dims = 512
batch_size = len(keep_strains)

hdf5_filename = f"{args["dataloader"][:args["dataloader"].find('.pth')]}.hdf5"


if args["dataloader_exists"]:
    
    if hdf5_filename not in os.listdir(home_dir):
        print(f"Loading data from {args["dataloader"]}...")
        dataloader_start = time.time()
        dataloader = torch.load(args["dataloader"], weights_only=True)
        print(f"Loading data took {time.time() - dataloader_start} seconds")
        print()

        print("Writing file to hdf5...")
        convert_to_hdf5_time = time.time()
        save_dataloader_to_hdf5(dataloader, hdf5_filename, classes)
        print(f"Converting .pth file to .hdf5 took {time.time() - convert_to_hdf5_time} seconds")

        print()

elif not args["dataloader_exists"]:
    print(f"Loading data into {args["dataloader"]}...")
    dataloader_start = time.time()
    data_dict = build_data_dict(home_dir, 25, keep_strains)
    dataset = DictionaryDataset(data_dict)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)


    print(f"Loading data took {time.time() - dataloader_start} seconds")
    print()

    print("Saving dataloader...")
    dataloader_save_start = time.time()
    torch.save(dataloader, args["dataloader"])
    print(f"Saving dataloader took {time.time() - dataloader_save_start} seconds")
    print()

    print("Writing file to hdf5...")
    print(f"Loading data from {args["dataloader"]}...")
    dataloader_start = time.time()
    dataloader = torch.load(args["dataloader"], weights_only=False)
    print(f"Loading data took {time.time() - dataloader_start} seconds")
    print()

    
    convert_to_hdf5_time = time.time()
    save_dataloader_to_hdf5(dataloader, hdf5_filename)
    print(f"Converting .pth file to .hdf5 took {time.time() - convert_to_hdf5_time} seconds")

    print()


if args["weights_exists"]:
    print(f"Loading weights from {args["weights"]}...")
    model_load_start = time.time()
    model_state_dict = torch.load(args["weights"], weights_only=False)
    model = vit_b_32(weights=None)
    vision_transformer = ViT(model).to(device)
    vision_transformer.load_state_dict(model_state_dict)
    print(f"Loading finetuned model took {time.time() - model_load_start} seconds")

    model_eval_start = time.time()
    print("Evaluating model...")
    embeddings, labels = eval_model(vision_transformer, hdf5_filename, classes)
    print(f"Evaluating model took {time.time() - model_eval_start} seconds")

    output_save_start = time.time()
    print("Saving output embeddings and labels...")
    np.save("output/embeddings.npy", embeddings)
    np.save("output/labels.npy", labels)
    print(f"Saving output embeddings and labels took {time.time() - output_save_start} seconds")

    print()

elif not args["weights_exists"]:
    print(f"Loading pretrained weights from IMAGENET1K...")
    model_load_start = time.time()
    model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    # for param in model.encoder.layers[:22].parameters():
    #     param.requires_grad = False
    # for param in model.encoder.layers[22:].parameters():
    #     param.requires_grad = True
    vision_transformer = ViT(model).to(device)
    print(f"Loading pre-trained VIT model took {time.time() - model_load_start} seconds")
    print()
    
    for param in vision_transformer.parameters():
        param.requires_grad = True
    # for param in vision_transformer.projection.parameters():
        # param.requires_grad = True

    print(f"Finetuning model...")
    print("Initial mem allocated", torch.cuda.memory_allocated())
    finetune_start = time.time()
    optimizer = torch.optim.Adam(vision_transformer.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = SimCLR()
    train_model(vision_transformer, optimizer, criterion, hdf5_filename, num_epochs=num_epochs, batch_size=batch_size)
    print(f"Finetuning model took {time.time() - finetune_start} seconds")
    print()
    
    print("Saving finetuned model weights...")
    weights_save_start = time.time()
    torch.save(vision_transformer.state_dict(), args["weights"])
    print(f"Saving model weights took {time.time() - weights_save_start} seconds")
    print()





exit()

embeddings = []
embeddings_start = time.time()


with torch.no_grad():
    for i in range(0, len(images)):
        processed = preprocess(Image.fromarray(images[i])).unsqueeze(0).to(device)
        print(type(processed))
        exit()
        image_features = model.encode_image(processed).cpu().numpy()[0]
        embeddings.append(image_features)
        
        

embeddings_end = time.time()
embeddings = np.array(embeddings)
print(f"{embeddings_dir}/{file[:len(file) - 4]}.npy")
np.save(f"{embeddings_dir}/plate_{plate_index}_{file[:len(file) - 4]}.npy", embeddings)

print(f"Generating embeddings for {file_path} took {embeddings_end - embeddings_start} seconds")
print()