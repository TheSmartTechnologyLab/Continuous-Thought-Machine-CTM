import torch
import torch.nn as nn
import torch.nn.functional as F # Used for GLU
import math
import numpy as np


class SynapseUNET(nn.Module):
    """
    UNET-style architecture for the Synapse Model (f_theta1 in the paper).

    This module implements the connections between neurons in the CTM's latent
    space. It processes the combined input (previous post-activation state z^t
    and attention output o^t) to produce the pre-activations (a^t) for the
    next internal tick (Eq. 1 in the paper).

    While a simpler Linear or MLP layer can be used, the paper notes
    that this U-Net structure empirically performed better, suggesting benefit
    from more flexible synaptic connections[cite: 79, 80]. This implementation
    uses `depth` points in linspace and creates `depth-1` down/up blocks.

    Args:
      in_dims (int): Number of input dimensions (d_model + d_input).
      out_dims (int): Number of output dimensions (d_model).
      depth (int): Determines structure size; creates `depth-1` down/up blocks.
      minimum_width (int): Smallest channel width at the U-Net bottleneck.
      dropout (float): Dropout rate applied within down/up projections.
    """
    def __init__(self,
                 out_dims,
                 depth,
                 minimum_width=16,
                 dropout=0.0):
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth # Store depth just for reference if needed

        # Define UNET structure based on depth
        # Creates `depth` width values, leading to `depth-1` blocks
        widths = np.linspace(out_dims, minimum_width, depth)

        # Initial projection layer
        self.first_projection = nn.Sequential(
            nn.LazyLinear(int(widths[0])), # Project to the first width
            nn.LayerNorm(int(widths[0])),
            nn.SiLU()
        )

        # Downward path (encoding layers)
        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()
        num_blocks = len(widths) - 1 # Number of down/up blocks created

        for i in range(num_blocks):
            # Down block: widths[i] -> widths[i+1]
            self.down_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i]), int(widths[i+1])),
                nn.LayerNorm(int(widths[i+1])),
                nn.SiLU()
            ))
            # Up block: widths[i+1] -> widths[i]
            # Note: Up blocks are added in order matching down blocks conceptually,
            # but applied in reverse order in the forward pass.
            self.up_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i+1]), int(widths[i])),
                nn.LayerNorm(int(widths[i])),
                nn.SiLU()
            ))
            # Skip connection LayerNorm operates on width[i]
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        # Initial projection
        out_first = self.first_projection(x)

        # Downward path, storing outputs for skip connections
        outs_down = [out_first]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))
        # outs_down contains [level_0, level_1, ..., level_depth-1=bottleneck] outputs

        # Upward path, starting from the bottleneck output
        outs_up = outs_down[-1] # Bottleneck activation
        num_blocks = len(self.up_projections) # Should be depth - 1

        for i in range(num_blocks):
            # Apply up projection in reverse order relative to down blocks
            # up_projection[num_blocks - 1 - i] processes deeper features first
            up_layer_idx = num_blocks - 1 - i
            out_up = self.up_projections[up_layer_idx](outs_up)

            # Get corresponding skip connection from downward path
            # skip_connection index = num_blocks - 1 - i (same as up_layer_idx)
            # This matches the output width of the up_projection[up_layer_idx]
            skip_idx = up_layer_idx
            skip_connection = outs_down[skip_idx]

            # Add skip connection and apply LayerNorm corresponding to this level
            # skip_lns index also corresponds to the level = skip_idx
            outs_up = self.skip_lns[skip_idx](out_up + skip_connection)

        # The final output after all up-projections
        return outs_up