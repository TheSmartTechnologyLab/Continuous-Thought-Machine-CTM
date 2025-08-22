"""Configuration module for CTM.

This module provides centralized configuration management for the CTM pipeline,
including device settings, model configurations, and training parameters.
"""

import torch
import os
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class ModelConfig:
    d_model: int = 128
    d_input: int = 48
    heads: int = 2
    n_synch_out: int = 32
    n_synch_action: int = 32
    synapse_depth: int = 3
    memory_length: int = 15
    memory_hidden_dims: int = 8
    dropout: float = 0.1
    deep_nlms: bool = False

@dataclass
class TrainingConfig:
    batch_size: int = 64
    iterations: int = 2000
    test_every: int = 100
    learning_rate: float = 1e-4
    checkpoint_dir: Optional[str] = None
    resume: bool = False
    make_gif: bool = False

class DeviceManager:
    """Manages device configuration and data transfer operations."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = self._setup_device(device)
        self.device_name = self.device.type
        self.n_gpu = torch.cuda.device_count() if self.device_name == 'cuda' else 0
    
    @staticmethod
    def _setup_device(device: Optional[str] = None) -> torch.device:
        """Setup the device for computation."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if isinstance(device, str):
            device = torch.device(device)
        
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            device = torch.device('cpu')
        
        return device
    
    def to_device(self, data: Union[torch.Tensor, List[torch.Tensor]]):
        """Move data to the configured device."""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device)
    
    def optimize_cuda_performance(self):
        """Configure CUDA settings for optimal performance."""
        if self.device_name == 'cuda':
            # Set CUDA optimization flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory management
            torch.cuda.empty_cache()
    
    def get_device_info(self) -> dict:
        """Get information about the current device configuration."""
        info = {
            'device_type': self.device_name,
            'n_gpu': self.n_gpu,
        }
        
        if self.device_name == 'cuda':
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'memory_allocated': f"{torch.cuda.memory_allocated(0)/1024**2:.2f} MB",
                'memory_cached': f"{torch.cuda.memory_reserved(0)/1024**2:.2f} MB"
            })
        
        return info

def get_default_device() -> DeviceManager:
    """Get the default device manager instance."""
    return DeviceManager()

def configure_training(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    return TrainingConfig(
        batch_size=args.batch_size,
        iterations=args.iterations,
        test_every=args.test_every,
        learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 1e-4,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        make_gif=args.make_gif if hasattr(args, 'make_gif') else False
    )

def configure_model(args) -> ModelConfig:
    """Create model configuration from arguments."""
    return ModelConfig(
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        memory_hidden_dims=args.memory_hidden_dims,
        dropout=args.dropout,
        deep_nlms=args.deep_nlms if hasattr(args, 'deep_nlms') else False
    )
