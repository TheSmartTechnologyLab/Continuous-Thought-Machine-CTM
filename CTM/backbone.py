import torch.nn as nn
from .model.squeeze import Identity, Squeeze
import torch
import torch.nn as nn
import torch.nn.functional as F # Used for GLU
import math
import numpy as np


# --- Backbone Modules ---

class ParityBackbone(nn.Module):
    def __init__(self, n_embeddings, d_embedding):
        super(ParityBackbone, self).__init__()
        self.embedding = nn.Embedding(n_embeddings, d_embedding)

    def forward(self, x):
        """
        Maps -1 (negative parity) to 0 and 1 (positive) to 1
        """
        x = (x == 1).long()
        return self.embedding(x.long()).transpose(1, 2) # Transpose for compatibility with other backbones

class QAMNISTOperatorEmbeddings(nn.Module):
    def __init__(self, num_operator_types, d_projection):
        super(QAMNISTOperatorEmbeddings, self).__init__()
        self.embedding = nn.Embedding(num_operator_types, d_projection)

    def forward(self, x):
        # -1 for plus and -2 for minus
        return self.embedding(-x - 1)

class QAMNISTIndexEmbeddings(torch.nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim

        embedding = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('embedding', embedding)

    def forward(self, x):
        return self.embedding[x]

class ThoughtSteps:
    """
    Helper class for managing "thought steps" in the ctm_qamnist pipeline.

    Args:
        iterations_per_digit (int): Number of iterations for each digit.
        iterations_per_question_part (int): Number of iterations for each question part.
        total_iterations_for_answering (int): Total number of iterations for answering.
        total_iterations_for_digits (int): Total number of iterations for digits.
        total_iterations_for_question (int): Total number of iterations for question.
    """
    def __init__(self, iterations_per_digit, iterations_per_question_part, total_iterations_for_answering, total_iterations_for_digits, total_iterations_for_question):
        self.iterations_per_digit = iterations_per_digit
        self.iterations_per_question_part = iterations_per_question_part
        self.total_iterations_for_digits = total_iterations_for_digits
        self.total_iterations_for_question = total_iterations_for_question
        self.total_iterations_for_answering = total_iterations_for_answering
        self.total_iterations = self.total_iterations_for_digits + self.total_iterations_for_question + self.total_iterations_for_answering

    def determine_step_type(self, stepi: int):
        is_digit_step = stepi < self.total_iterations_for_digits
        is_question_step = self.total_iterations_for_digits <= stepi < self.total_iterations_for_digits + self.total_iterations_for_question
        is_answer_step = stepi >= self.total_iterations_for_digits + self.total_iterations_for_question
        return is_digit_step, is_question_step, is_answer_step

    def determine_answer_step_type(self, stepi: int):
        step_within_questions = stepi - self.total_iterations_for_digits
        if step_within_questions % (2 * self.iterations_per_question_part) < self.iterations_per_question_part:
            is_index_step = True
            is_operator_step = False
        else:
            is_index_step = False
            is_operator_step = True
        return is_index_step, is_operator_step

class MNISTBackbone(nn.Module):
    """
    Simple backbone for MNIST feature extraction.
    """
    def __init__(self, d_input):
        super(MNISTBackbone, self).__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.layers(x)


class MiniGridBackbone(nn.Module):
    def __init__(self, d_input, grid_size=7, num_objects=11, num_colors=6, num_states=3, embedding_dim=8):
        super().__init__()
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        self.state_embedding = nn.Embedding(num_states, embedding_dim)

        self.position_embedding = nn.Embedding(grid_size * grid_size, embedding_dim)

        self.project_to_d_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.Linear(d_input, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        x = x.long()
        B, H, W, C = x.size()

        object_idx = x[:,:,:, 0]
        color_idx =  x[:,:,:, 1]
        state_idx =  x[:,:,:, 2]

        obj_embed = self.object_embedding(object_idx)
        color_embed = self.color_embedding(color_idx)
        state_embed = self.state_embedding(state_idx)

        pos_idx = torch.arange(H * W, device=x.device).view(1, H, W).expand(B, -1, -1)
        pos_embed = self.position_embedding(pos_idx)

        out = self.project_to_d_projection(torch.cat([obj_embed, color_embed, state_embed, pos_embed], dim=-1))
        return out

class ClassicControlBackbone(nn.Module):
    def __init__(self, d_input):
        super().__init__()
        self.input_projector = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        return self.input_projector(x)


class ShallowWide(nn.Module):
    """
    Simple, wide, shallow convolutional backbone for image feature extraction.

    Alternative to ResNet, uses grouped convolutions and GLU activations.
    Fixed structure, useful for specific experiments.
    """
    def __init__(self):
        super(ShallowWide, self).__init__()
        # LazyConv2d infers input channels
        self.layers = nn.Sequential(
            nn.LazyConv2d(4096, kernel_size=3, stride=2, padding=1), # Output channels = 4096
            nn.GLU(dim=1), # Halves channels to 2048
            nn.BatchNorm2d(2048),
            # Grouped convolution maintains width but processes groups independently
            nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1, groups=32),
            nn.GLU(dim=1), # Halves channels to 2048
            nn.BatchNorm2d(2048)
        )
    def forward(self, x):
        return self.layers(x)


class PretrainedResNetWrapper(nn.Module):
    """
    Wrapper to use standard pre-trained ResNet models from torchvision.

    Loads a specified ResNet architecture pre-trained on ImageNet, removes the
    final classification layer (fc), average pooling, and optionally later layers
    (e.g., layer4), allowing it to be used as a feature extractor backbone.

    Args:
        resnet_type (str): Name of the ResNet model (e.g., 'resnet18', 'resnet50').
        fine_tune (bool): If False, freezes the weights of the pre-trained backbone.
    """
    def __init__(self, resnet_type, fine_tune=True):
        super(PretrainedResNetWrapper, self).__init__()
        self.resnet_type = resnet_type
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', resnet_type, pretrained=True)

        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remove final layers to use as feature extractor
        self.backbone.avgpool = Identity()
        self.backbone.fc = Identity()
        # Keep layer4 by default, user can modify instance if needed
        # self.backbone.layer4 = Identity()

    def forward(self, x):
        # Get features from the modified ResNet
        out = self.backbone(x)

        # Reshape output to (B, C, H, W) - This is heuristic based on original comment.
        # User might need to adjust this based on which layers are kept/removed.
        # Infer C based on ResNet type (example values)
        nc = 256 if ('18' in self.resnet_type or '34' in self.resnet_type) else 512 if '50' in self.resnet_type else 1024 if '101' in self.resnet_type else 2048 # Approx for layer3/4 output channel numbers
        # Infer H, W assuming output is flattened C * H * W
        num_features = out.shape[-1]
        # This calculation assumes nc is correct and feature map is square
        wh_squared = num_features / nc
        if wh_squared < 0 or not float(wh_squared).is_integer():
             print(f"Warning: Cannot reliably reshape PretrainedResNetWrapper output. nc={nc}, num_features={num_features}")
             # Return potentially flattened features if reshape fails
             return out
        wh = int(np.sqrt(wh_squared))

        return out.reshape(x.size(0), nc, wh, wh)