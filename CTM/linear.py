import torch
import torch.nn as nn
import torch.nn.functional as F # Used for GLU
import math
import numpy as np
from .squeeze import Identity


class SuperLinear(nn.Module):
    """
    SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM.

    This layer is the core component enabling Neuron-Level Models (NLMs),
    referred to as g_theta_d in the paper (Eq. 3). It applies N independent
    linear transformations (or small MLPs when used sequentially) to corresponding
    slices of the input tensor along a specified dimension (typically the neuron
    or feature dimension).

    How it works for NLMs:
    - The input `x` is expected to be the pre-activation history for each neuron,
      shaped (batch_size, n_neurons=N, history_length=in_dims).
    - This layer holds unique weights (`w1`) and biases (`b1`) for *each* of the `N` neurons.
      `w1` has shape (in_dims, out_dims, N), `b1` has shape (1, N, out_dims).
    - `torch.einsum('bni,iog->bno', x, self.w1)` performs N independent matrix
      multiplications in parallel (mapping from dim `i` to `o` for each neuron `n`):
        - For each neuron `n` (from 0 to N-1):
        - It takes the neuron's history `x[:, n, :]` (shape B, in_dims).
        - Multiplies it by the neuron's unique weight matrix `self.w1[:, :, n]` (shape in_dims, out_dims).
        - Resulting in `out[:, n, :]` (shape B, out_dims).
    - The unique bias `self.b1[:, n, :]` is added.
    - The result is squeezed on the last dim (if out_dims=1) and scaled by `T`.

    This allows each neuron `d` to process its temporal history `A_d^t` using
    its private parameters `theta_d` to produce the post-activation `z_d^{t+1}`,
    enabling the fine-grained temporal dynamics central to the CTM[cite: 7, 30, 85].
    It's typically used within the `trace_processor` module of the main CTM class.

    Args:
      in_dims (int): Input dimension (typically `memory_length`).
      out_dims (int): Output dimension per neuron.
      N (int): Number of independent linear models (typically `d_model`).
      T (float): Initial value for learnable temperature/scaling factor applied to output.
      do_norm (bool): Apply Layer Normalization to the input history before linear transform.
      dropout (float): Dropout rate applied to the input.
    """
    def __init__(self,
                 in_dims,
                 out_dims,
                 N,
                 T=1.0,
                 do_norm=False,
                 dropout=0):
        super().__init__()
        # N is the number of neurons (d_model), in_dims is the history length (memory_length)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.in_dims = in_dims # Corresponds to memory_length
        # LayerNorm applied across the history dimension for each neuron independently
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm

        # Initialize weights and biases
        # w1 shape: (memory_length, out_dims, d_model)
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        # b1 shape: (1, d_model, out_dims)
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        # Learnable temperature/scaler T
        self.register_parameter('T', nn.Parameter(torch.Tensor([T])))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, expected shape (B, N, in_dims)
                              where B=batch, N=d_model, in_dims=memory_length.
        Returns:
            torch.Tensor: Output tensor, shape (B, N) after squeeze(-1).
        """
        # Input shape: (B, D, M) where D=d_model=N neurons in CTM, M=history/memory length
        out = self.dropout(x)
        # LayerNorm across the memory_length dimension (dim=-1)
        out = self.layernorm(out) # Shape remains (B, N, M)

        # Apply N independent linear models using einsum
        # einsum('BDM,MHD->BDH', ...)
        # x: (B=batch size, D=N neurons, one NLM per each of these, M=history/memory length)
        # w1: (M, H=hidden dims if using MLP, otherwise output, D=N neurons, parallel)
        # b1: (1, D=N neurons, H)
        # einsum result: (B, D, H)
        # Applying bias requires matching shapes, b1 is broadcasted.
        out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1

        # Squeeze the output dimension (assumed to be 1 usually) and scale by T
        # This matches the original code's structure exactly.
        out = out.squeeze(-1) / self.T
        return out