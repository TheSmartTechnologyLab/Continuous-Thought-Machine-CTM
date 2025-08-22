"""Pipeline entrypoint to run CTM experiments.

This script provides a single entrypoint to prepare datasets (CIFAR10, CIFAR100, MNIST, SVHN),
construct a ContinuousThoughtMachine with a selectable backbone (existing resnet variants or
popular torchvision backbones such as mobilenet_v2 and efficientnet_b0), run a training loop,
evaluate, and optionally produce an animated GIF of model dynamics.

Usage (from repo root):
  python -m CTM.model.pipeline --dataset cifar10 --backbone resnet18-2 --iterations 2000
"""

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
import numpy as np
import math
import warnings
import os

# Local imports
from .model import ContinuousThoughtMachine
from .pretrain import prepare_resnet_backbone, TorchvisionBackboneWrapper
from .train import train_ctm as train_loop
from .gif import make_gif
from .utils import (
    get_loss,
    calculate_accuracy,
    save_checkpoint_local,
    save_checkpoint_to_hf,
    load_checkpoint,
    get_latest_checkpoint,
)


# Use get_loss and calculate_accuracy from model.utils


def prepare_data(dataset_name, batch_size=64):
    dataset_name = dataset_name.lower()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == 'cifar10':
        train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        out_dims = 10
    elif dataset_name == 'cifar100':
        train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        out_dims = 100
    elif dataset_name == 'mnist':
        train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        out_dims = 10
    elif dataset_name == 'svhn':
        train = datasets.SVHN(root="./data", split='train', download=True, transform=transform)
        test = datasets.SVHN(root="./data", split='test', download=True, transform=transform)
        out_dims = 10
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    return trainloader, testloader, out_dims



def build_model(args, device, out_dims):
    # Build CTM with backbone_type 'none' first so we can inject custom backbones if needed
    model = ContinuousThoughtMachine(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_nlms,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_layernorm_nlm,
        backbone_type='none',
        positional_embedding_type='none',
        out_dims=out_dims,
        prediction_reshaper=[-1],
        dropout=args.dropout,
        neuron_select_type=args.neuron_select_type,
        n_random_pairing_self=args.n_random_pairing_self,
    ).to(device)

    # Attach backbone
    backbone_name = args.backbone.lower()
    if backbone_name.startswith('resnet'):
        # Use existing prepare_resnet_backbone helper
        model.backbone = prepare_resnet_backbone(backbone_name)
    else:
        # Try torchvision family
        try:
            model.backbone = TorchvisionBackboneWrapper(backbone_name, pretrained=args.pretrained, device=device)
        except Exception as e:
            warnings.warn(f'Failed to create torchvision wrapper for {backbone_name}: {e}. Falling back to identity backbone.')
            model.backbone = nn.Identity()

    # Set positional embedding to no-op for non-resnet backbones
    model.positional_embedding = (lambda x: 0)

    # Kv and q projections and attention were set in __init__ only if heads>0; ensure they exist
    if model.kv_proj is None and args.heads:
        model.kv_proj = nn.Sequential(nn.LazyLinear(model.d_input), nn.LayerNorm(model.d_input))
    if model.q_proj is None and args.heads:
        model.q_proj = nn.LazyLinear(model.d_input)
    if model.attention is None and args.heads:
        model.attention = nn.MultiheadAttention(model.d_input, args.heads, batch_first=True)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'svhn'])
    parser.add_argument('--backbone', default='resnet18-2')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_input', type=int, default=48)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--synapse_depth', type=int, default=3)
    parser.add_argument('--memory_length', type=int, default=15)
    parser.add_argument('--deep_nlms', action='store_true')
    parser.add_argument('--memory_hidden_dims', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--neuron_select_type', default='random-pairing')
    parser.add_argument('--n_random_pairing_self', type=int, default=0)
    parser.add_argument('--do_layernorm_nlm', action='store_true', help='Use layernorm in NLM modules')
    parser.add_argument('--make_gif', action='store_true')
    parser.add_argument('--checkpoint_dir', default=None, help='Local directory to save checkpoints')
    parser.add_argument('--hf_repo', default=None, help='Hugging Face repo id to upload checkpoints (e.g., username/repo)')
    parser.add_argument('--hf_token', default=None, help='Hugging Face token (or use env HF_TOKEN)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest local checkpoint in checkpoint_dir')
    args = parser.parse_args()

    device = torch.device(args.device)
    trainloader, testloader, out_dims = prepare_data(args.dataset, batch_size=args.batch_size)

    model = build_model(args, device, out_dims)

    model = train_loop(
        model,
        trainloader,
        testloader,
        device,
        iterations=args.iterations,
        checkpoint_dir=args.checkpoint_dir,
        hf_repo=args.hf_repo,
        hf_token=(args.hf_token or os.environ.get('HF_TOKEN')),
        resume=args.resume,
        args=args,
    )

    # Quick evaluation and optional gif
    model.eval()
    with torch.inference_mode():
        inputs, targets = next(iter(testloader))
        inputs = inputs.to(device)
        predictions, certainties, (synch_out_tracking, synch_action_tracking), pre_acts, post_acts, attention = model(inputs, track=True)

        print('Sample prediction produced; optional GIF generation follows if requested.')

        if args.make_gif:
            filename = f"ctm_output_{args.dataset}_{args.backbone}.gif"
            make_gif(
                predictions.detach().cpu().numpy(),
                certainties.detach().cpu().numpy(),
                targets.detach().cpu().numpy(),
                pre_acts,
                post_acts,
                attention,
                inputs.detach().cpu().numpy(),
                filename,
            )


if __name__ == '__main__':
    main()