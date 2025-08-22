import torch
import torch.nn as nn
import torch.nn.functional as F # Used for GLU
import math
import numpy as np


class Identity(nn.Module):
    """
    Identity Module.

    Returns the input tensor unchanged. Useful as a placeholder or a no-op layer
    in nn.Sequential containers or conditional network parts.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Squeeze(nn.Module):
    """
    Squeeze Module.

    Removes a specified dimension of size 1 from the input tensor.
    Useful for incorporating tensor dimension squeezing within nn.Sequential.

    Args:
      dim (int): The dimension to squeeze.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)