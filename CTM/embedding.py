import torch
import torch.nn as nn
import torch.nn.functional as F # Used for GLU
import math
import numpy as np


# --- Positional Encoding Modules ---

class LearnableFourierPositionalEncoding(nn.Module):
    """
    Learnable Fourier Feature Positional Encoding.

    Implements Algorithm 1 from "Learnable Fourier Features for Multi-Dimensional
    Spatial Positional Encoding" (https://arxiv.org/pdf/2106.02795.pdf).
    Provides positional information for 2D feature maps.

    Args:
        d_model (int): The output dimension of the positional encoding (D).
        G (int): Positional groups (default 1).
        M (int): Dimensionality of input coordinates (default 2 for H, W).
        F_dim (int): Dimension of the Fourier features.
        H_dim (int): Hidden dimension of the MLP.
        gamma (float): Initialization scale for the Fourier projection weights (Wr).
    """
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma=1/2.5,
                 ):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = d_model
        self.gamma = gamma

        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GLU(), # Halves H_dim
            nn.Linear(self.H_dim // 2, self.D // self.G),
            nn.LayerNorm(self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Computes positional encodings for the input feature map x.

        Args:
            x (torch.Tensor): Input feature map, shape (B, C, H, W).

        Returns:
            torch.Tensor: Positional encoding tensor, shape (B, D, H, W).
        """
        B, C, H, W = x.shape
        # Creates coordinates based on (H, W) and repeats for batch B.
        # Takes x[:,0] assuming channel dim isn't needed for coords.
        x_coord = add_coord_dim(x[:,0]) # Expects (B, H, W) -> (B, H, W, 2)

        # Compute Fourier features
        projected = self.Wr(x_coord) # (B, H, W, F_dim // 2)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = (1.0 / math.sqrt(self.F_dim)) * torch.cat([cosines, sines], dim=-1) # (B, H, W, F_dim)

        # Project features through MLP
        Y = self.mlp(F) # (B, H, W, D // G)

        # Reshape to (B, D, H, W)
        PEx = Y.permute(0, 3, 1, 2) # Assuming G=1
        return PEx


class MultiLearnableFourierPositionalEncoding(nn.Module):
    """
    Combines multiple LearnableFourierPositionalEncoding modules with different
    initialization scales (gamma) via a learnable weighted sum.

    Allows the model to learn an optimal combination of positional frequencies.

    Args:
        d_model (int): Output dimension of the encoding.
        G, M, F_dim, H_dim: Parameters passed to underlying LearnableFourierPositionalEncoding.
        gamma_range (list[float]): Min and max gamma values for the linspace.
        N (int): Number of parallel embedding modules to create.
    """
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma_range=[1.0, 0.1], # Default range
                 N=10,
                 ):
        super().__init__()
        self.embedders = nn.ModuleList()
        for gamma in np.linspace(gamma_range[0], gamma_range[1], N):
            self.embedders.append(LearnableFourierPositionalEncoding(d_model, G, M, F_dim, H_dim, gamma))

        # Renamed parameter from 'combination' to 'combination_weights' for clarity only in comments
        # Actual registered name remains 'combination' as in original code
        self.register_parameter('combination', torch.nn.Parameter(torch.ones(N), requires_grad=True))
        self.N = N


    def forward(self, x):
        """
        Computes combined positional encoding.

        Args:
            x (torch.Tensor): Input feature map, shape (B, C, H, W).

        Returns:
            torch.Tensor: Combined positional encoding tensor, shape (B, D, H, W).
        """
        # Compute embeddings from all modules and stack: (N, B, D, H, W)
        pos_embs = torch.stack([emb(x) for emb in self.embedders], dim=0)

        # Compute combination weights using softmax
        # Use registered parameter name 'combination'
        # Reshape weights for broadcasting: (N,) -> (N, 1, 1, 1, 1)
        weights = F.softmax(self.combination, dim=-1).view(self.N, 1, 1, 1, 1)

        # Compute weighted sum over the N dimension
        combined_emb = (pos_embs * weights).sum(0) # (B, D, H, W)
        return combined_emb


class CustomRotationalEmbedding(nn.Module):
    """
    Custom Rotational Positional Embedding.

    Generates 2D positional embeddings based on rotating a fixed start vector.
    The rotation angle for each grid position is determined primarily by its
    horizontal position (width dimension). The resulting rotated vectors are
    concatenated and projected.

    Note: The current implementation derives angles only from the width dimension (`x.size(-1)`).

    Args:
        d_model (int): Dimensionality of the output embeddings.
    """
    def __init__(self, d_model):
        super(CustomRotationalEmbedding, self).__init__()
        # Learnable 2D start vector
        self.register_parameter('start_vector', nn.Parameter(torch.Tensor([0, 1]), requires_grad=True))
        # Projects the 4D concatenated rotated vectors to d_model
        # Input size 4 comes from concatenating two 2D rotated vectors
        self.projection = nn.Sequential(nn.Linear(4, d_model))

    def forward(self, x):
        """
        Computes rotational positional embeddings based on input width.

        Args:
            x (torch.Tensor): Input tensor (used for shape and device),
                              shape (batch_size, channels, height, width).
        Returns:
            Output tensor containing positional embeddings,
            shape (1, d_model, height, width) - Batch dim is 1 as PE is same for all.
        """
        B, C, H, W = x.shape
        device = x.device

        # --- Generate rotations based only on Width ---
        # Angles derived from width dimension
        theta_rad = torch.deg2rad(torch.linspace(0, 180, W, device=device)) # Angle per column
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)

        # Create rotation matrices: Shape (W, 2, 2)
        # Use unsqueeze(1) to allow stacking along dim 1
        rotation_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1), # Shape (W, 2)
            torch.stack([sin_theta, cos_theta], dim=-1)  # Shape (W, 2)
        ], dim=1) # Stacks along dim 1 -> Shape (W, 2, 2)

        # Rotate the start vector by column angle: Shape (W, 2)
        rotated_vectors = torch.einsum('wij,j->wi', rotation_matrices, self.start_vector)

        # --- Create Grid Key ---
        # Original code uses repeats based on rotated_vectors.shape[0] (which is W) for both dimensions.
        # This creates a (W, W, 4) key tensor.
        key = torch.cat((
            torch.repeat_interleave(rotated_vectors.unsqueeze(1), W, dim=1), # (W, 1, 2) -> (W, W, 2)
            torch.repeat_interleave(rotated_vectors.unsqueeze(0), W, dim=0)  # (1, W, 2) -> (W, W, 2)
        ), dim=-1) # Shape (W, W, 4)

        # Project the 4D key vector to d_model: Shape (W, W, d_model)
        pe_grid = self.projection(key)

        # Reshape to (1, d_model, W, W) and then select/resize to target H, W?
        # Original code permutes to (d_model, W, W) and unsqueezes to (1, d_model, W, W)
        pe = pe_grid.permute(2, 0, 1).unsqueeze(0)

        # If H != W, this needs adjustment. Assuming H=W or cropping/padding happens later.
        # Let's return the (1, d_model, W, W) tensor as generated by the original logic.
        # If H != W, downstream code must handle the mismatch or this PE needs modification.
        if H != W:
            # Simple interpolation/cropping could be added, but sticking to original logic:
            # Option 1: Interpolate
            # pe = F.interpolate(pe, size=(H, W), mode='bilinear', align_corners=False)
            # Option 2: Crop/Pad (e.g., crop if W > W_target, pad if W < W_target)
            # Sticking to original: return shape (1, d_model, W, W)
            pass

        return pe

class CustomRotationalEmbedding1D(nn.Module):
    def __init__(self, d_model):
        super(CustomRotationalEmbedding1D, self).__init__()
        self.projection = nn.Linear(2, d_model)

    def forward(self, x):
        start_vector = torch.tensor([0., 1.], device=x.device, dtype=torch.float)
        theta_rad = torch.deg2rad(torch.linspace(0, 180, x.size(2), device=x.device))
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_theta = cos_theta.unsqueeze(1)  # Shape: (height, 1)
        sin_theta = sin_theta.unsqueeze(1)  # Shape: (height, 1)

        # Create rotation matrices
        rotation_matrices = torch.stack([
        torch.cat([cos_theta, -sin_theta], dim=1),
        torch.cat([sin_theta, cos_theta], dim=1)
        ], dim=1)  # Shape: (height, 2, 2)

        # Rotate the start vector
        rotated_vectors = torch.einsum('bij,j->bi', rotation_matrices, start_vector)

        pe = self.projection(rotated_vectors)
        pe = torch.repeat_interleave(pe.unsqueeze(0), x.size(0), 0)
        return pe.transpose(1, 2) # Transpose for compatibility with other backbones
