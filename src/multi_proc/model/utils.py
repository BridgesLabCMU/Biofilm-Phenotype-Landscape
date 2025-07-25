import torch
import os
from torch import nn
import torch.nn.functional as F
gpu_count = torch.cuda.device_count()
if gpu_count > 1:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    device=rank
elif gpu_count == 1:
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device}")
