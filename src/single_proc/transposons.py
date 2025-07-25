import time
start = time.time()
import cv2
import os
import platform
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
from natsort import natsorted
import re
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Loading model took {time.time() - start} seconds")

home_dir = "Embeddings"

dir = f"."
paths = os.listdir(dir)
folders = []
for entry in paths: 
    directory = f"{dir}/{entry}"
    if os.path.isdir(directory):
        if "Plate" in directory:
            folders.append(f"{dir}/{entry}")

images_dirs = []
for folder in natsorted(folders):
    for sub_folder in [f.path for f in os.scandir(folder) if f.is_dir()]:
        images_dirs.append(f"{sub_folder}")
print(images_dirs)

for dir in images_dirs:
    print(dir)
    print()

os.makedirs(f"{home_dir}", exist_ok=True)


index = 1
for dir in images_dirs:
    for file in natsorted(os.listdir(dir)):
        if file.find("mask") == -1 and file.find("Thumb") == -1 and file.find(".csv") == -1:
            magnification = ""
            if file.find("4x") > 0:
                magnification = "4x"
                continue
            elif file.find("03") > 0:
                magnification = "10x"
            elif file.find("20x") > 0:
                magnification = "20x"
                continue
            elif file.find("40x") > 0:
                magnification = "40x"
                continue
            print(index)
            index += 1

            plate_id = re.search(r"[P|p]late[0-9][_|(0-9)]", dir)
            if plate_id.group(0)[-1] == "_":
                plate_id = plate_id.group(0)[:-1]
            else:
                plate_id = plate_id.group(0)
            plate_id = plate_id[plate_id.find("e")+1:]
            well_id = file[:file.find("_")]
            if plate_id[0] == "0":
                plate_id = plate_id[1]

            print("MAGNIFICATION: ", magnification)
            if not os.path.exists(f"{home_dir}/{magnification}"):
                os.makedirs(f"{home_dir}/{magnification}")
            embeddings_dir = f"{home_dir}/{magnification}"
        
            image_stack = []
            file_path = f"{dir}/{file}"
            print(f"Storing image stack and computing embeddings for {file_path}")
            ret,images = cv2.imreadmulti(mats=image_stack,
                                         filename=file_path,
                                         start=0,
                                         count=31,
                                         flags=cv2.IMREAD_ANYCOLOR)
            embeddings = []
            embeddings_start = time.time()
            with torch.no_grad():
                for i in range(0, len(images)):
                    processed = preprocess(Image.fromarray(images[i])).unsqueeze(0).to(device)
                    image_features = model.encode_image(processed).cpu().numpy()[0]
                    embeddings.append(image_features)
            embeddings_end = time.time()
            embeddings = np.array(embeddings)
            print(f"{embeddings_dir}/{plate_id}_{well_id}.npy")
            np.save(f"{embeddings_dir}/{plate_id}_{well_id}.npy", embeddings)
            
            print(f"Generating embeddings for {file_path} took {embeddings_end - embeddings_start} seconds")
            print()
