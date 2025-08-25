"""
Continuous Thought Machine (CTM) Package

This package provides implementations for training and evaluating CTM models
on various datasets with different backbone architectures.
"""

# Core components
from .model import ContinuousThoughtMachine

# Main pipeline
from .enhanced_pipeline import main

__all__ = [
    'ContinuousThoughtMachine',
    'train_loop',
    'prepare_data',
    'build_model',
    'main',
]
from .train import train_ctm
from .config import DeviceManager, configure_training, configure_model

__version__ = "1.0.0"

# Make commonly used classes and functions available at package level
__all__ = [
    'ContinuousThoughtMachine',
    'train_loop',
    'prepare_data',
    'build_model',
    'train_ctm',
    'DeviceManager',
    'configure_training',
    'configure_model',
    'enhanced_main'
]
