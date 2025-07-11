import torch
from torch.utils.data import DistributedSampler

dataset = torch.load("../dataloaders/dataset.pth", weights_only=False)
print(dataset.shape)