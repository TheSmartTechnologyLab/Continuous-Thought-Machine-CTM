"""Enhanced pipeline for CTM training and evaluation.

This module provides a streamlined pipeline for training and evaluating CTM models,
with optimizations for both CPU and CUDA execution.
"""

import argparse
import torch
import torch.nn as nn
import os

from .config import DeviceManager, configure_training, configure_model
from .data import DataManager
from .train import train_ctm
from .model import ContinuousThoughtMachine
from .gif import make_gif

def setup_args():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Train CTM models with optimized pipeline')
    
    # Dataset and model configuration
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'svhn'],
                      help='Dataset to use for training')
    parser.add_argument('--backbone', default='resnet18-2',
                      help='Backbone architecture (e.g., resnet18-2, mobilenet_v2)')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained backbone')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--iterations', type=int, default=2000,
                      help='Number of training iterations')
    parser.add_argument('--test_every', type=int, default=100,
                      help='Evaluation frequency')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_input', type=int, default=48)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--synapse_depth', type=int, default=3)
    parser.add_argument('--memory_length', type=int, default=15)
    parser.add_argument('--memory_hidden_dims', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--deep_nlms', action='store_true')
    
    # Checkpoint and visualization
    parser.add_argument('--checkpoint_dir', default=None,
                      help='Directory for saving checkpoints')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from latest checkpoint')
    parser.add_argument('--make_gif', action='store_true',
                      help='Generate visualization GIF')
    
    # Device configuration
    parser.add_argument('--device', default=None,
                      help='Device to use (cuda/cpu). If not specified, uses CUDA if available.')
    
    return parser.parse_args()

def build_model(model_config, backbone_name, pretrained, device, out_dims):
    """Build CTM model with specified configuration."""
    model = ContinuousThoughtMachine(
        out_dims=out_dims,
        d_model=model_config.d_model,
        d_input=model_config.d_input,
        heads=model_config.heads,
        n_synch_out=model_config.n_synch_out,
        n_synch_action=model_config.n_synch_action,
        synapse_depth=model_config.synapse_depth,
        memory_length=model_config.memory_length,
        memory_hidden_dims=model_config.memory_hidden_dims,
        dropout=model_config.dropout,
        deep_nlms=model_config.deep_nlms
    ).to(device)
    
    # Configure backbone
    if backbone_name.startswith('resnet'):
        from .pretrain import prepare_resnet_backbone
        model.backbone = prepare_resnet_backbone(backbone_name)
    else:
        from .pretrain import TorchvisionBackboneWrapper
        try:
            model.backbone = TorchvisionBackboneWrapper(backbone_name, pretrained=pretrained, device=device)
        except Exception as e:
            warnings.warn(f'Failed to create torchvision wrapper for {backbone_name}: {e}. Using identity backbone.')
            model.backbone = nn.Identity()
    
    return model

def main():
    """Main entry point for the CTM pipeline."""
    # Parse arguments and setup configurations
    args = setup_args()
    
    # Setup device
    device_manager = DeviceManager(args.device)
    print(f"\nDevice configuration:")
    for k, v in device_manager.get_device_info().items():
        print(f"  {k}: {v}")
    
    # Configure training and model
    training_config = configure_training(args)
    model_config = configure_model(args)
    
    # Setup data
    data_manager = DataManager(batch_size=args.batch_size)
    trainloader, testloader, out_dims = data_manager.prepare_data(args.dataset)
    
    # Optimize data loading for device
    data_manager.optimize_memory_usage(device_manager.device)
    
    # Build and train model
    model = build_model(
        model_config,
        args.backbone,
        args.pretrained,
        device_manager.device,
        out_dims
    )
    
    print("\nStarting training...")
    model = train_ctm(
        model,
        trainloader,
        testloader,
        device_manager,
        training_config
    )
    
    # Generate visualization if requested
    if args.make_gif:
        print("\nGenerating visualization...")
        model.eval()
        with torch.inference_mode():
            inputs, targets = next(iter(testloader))
            inputs = inputs.to(device_manager.device)
            predictions, certainties, (synch_out_tracking, synch_action_tracking), pre_acts, post_acts, attention = model(
                inputs, track=True)
            
            filename = f"ctm_output_{args.dataset}_{args.backbone}.gif"
            make_gif(
                predictions.detach().cpu().numpy(),
                certainties.detach().cpu().numpy(),
                targets.numpy(),
                pre_acts,
                post_acts,
                attention,
                inputs.detach().cpu().numpy(),
                filename,
            )
            print(f"Visualization saved as {filename}")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
