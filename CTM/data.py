"""Data handling module for CTM.

This module provides utilities for loading and preprocessing data for the CTM pipeline.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any
import numpy as np

class DataManager:
    """Manages data loading and preprocessing for CTM."""
    
    SUPPORTED_DATASETS = {
        'cifar10': (datasets.CIFAR10, 10),
        'cifar100': (datasets.CIFAR100, 100),
        'mnist': (datasets.MNIST, 10),
        'svhn': (datasets.SVHN, 10)
    }
    
    def __init__(self, batch_size: int = 64, num_workers: int = 2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup data transforms for training and testing."""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def prepare_data(self, dataset_name: str) -> Tuple[DataLoader, DataLoader, int]:
        """
        Prepare training and testing data loaders.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            tuple: (train_loader, test_loader, number_of_classes)
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f'Unsupported dataset: {dataset_name}. '
                           f'Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}')
        
        dataset_class, num_classes = self.SUPPORTED_DATASETS[dataset_name]
        
        # Handle SVHN's different split naming
        if dataset_name == 'svhn':
            train_split, test_split = 'train', 'test'
            dataset_kwargs = {'split': train_split}
            test_kwargs = {'split': test_split}
        else:
            dataset_kwargs = {'train': True}
            test_kwargs = {'train': False}
        
        # Create datasets
        train_dataset = dataset_class(
            root="./data",
            download=True,
            transform=self.transform,
            **dataset_kwargs
        )
        
        test_dataset = dataset_class(
            root="./data",
            download=True,
            transform=self.transform,
            **test_kwargs
        )
        
        # Create data loaders with automatic num_workers selection
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, test_loader, num_classes
    
    @staticmethod
    def get_batch_statistics(batch: torch.Tensor) -> Dict[str, Any]:
        """
        Get statistics for a batch of data.
        
        Args:
            batch: Input batch tensor
            
        Returns:
            dict: Dictionary containing batch statistics
        """
        return {
            'mean': batch.mean().item(),
            'std': batch.std().item(),
            'min': batch.min().item(),
            'max': batch.max().item(),
            'shape': tuple(batch.shape)
        }
    
    def optimize_memory_usage(self, device):
        """Optimize memory usage based on device."""
        if device.type == 'cuda':
            # Optimize CUDA memory usage
            self.num_workers = 4  # Increase workers for GPU
            torch.cuda.empty_cache()
        else:
            # Optimize CPU memory usage
            self.num_workers = 2  # Reduce workers for CPU
