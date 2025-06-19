import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")