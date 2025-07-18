import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

gpu_count = int(os.environ['CUDA_VISIBLE_DEVICES'].split(",")[-1]) + 1
gpu_count = 8
if gpu_count > 1:
    local_rank = int(os.environ["LOCAL_RANK"])
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    device=rank
elif gpu_count == 1:
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device}")
