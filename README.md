# Continuous-Thought-Machine-CTM

Lightweight research code for the Continuous Thought Machine (CTM).

This repository contains the CTM model and a single, flexible pipeline to run experiments across common image datasets (CIFAR-10, CIFAR-100, MNIST, SVHN) and with multiple backbone choices (internal ResNet variants or torchvision models such as MobileNet/EfficientNet).

## Quick start

1. Create a virtual environment (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the pipeline help to see options:

```powershell
python -m CTM.model.pipeline --help
```

4. Run a short smoke test (10 iterations) on CIFAR-10 with the built-in ResNet backbone:

```powershell
python -m CTM.model.pipeline --dataset cifar10 --backbone resnet18-2 --iterations 10 --checkpoint_dir .\checkpoints
```

## Pipeline features

- Single entrypoint: `CTM/model/pipeline.py` (run as module with `python -m CTM.model.pipeline`).
- Supported datasets: `cifar10`, `cifar100`, `mnist`, `svhn`.
- Backbones:
	- Internal ResNet variants: `resnet18-1`, `resnet18-2`, `resnet34-2`, etc. (uses `CTM/model/pretrain.py`).
	- Torchvision models (best-effort wrapper): `mobilenet_v2`, `efficientnet_b0`, etc. The wrapper attempts to extract spatial feature maps; some models may require custom handling.
- Checkpointing: local checkpoint saving is automatic when `--checkpoint_dir` is provided. Optional upload to Hugging Face model repos is supported via `--hf_repo` and `--hf_token` (or env var `HF_TOKEN`).
- Resume: use `--resume` to pick up from the latest checkpoint in `--checkpoint_dir`.
- GIF visualization: use `--make_gif` to save a visualization of CTM internal dynamics for a sample batch.

## Important notes

- PyTorch wheels are platform/GPU-specific. If you need GPU support, install the correct `torch`/`torchvision` wheel for your CUDA version using PyTorch's official instructions. The `requirements.txt` contains minimum-version guidance but does not pin a CUDA-specific wheel.
- The torchvision backbone wrapper is heuristic and may not produce the expected feature map shape for every architecture; if you plan to use MobileNet/EfficientNet extensively I can add model-specific extraction logic.
- Hugging Face uploads require `huggingface-hub` and a valid token with write permissions for the target repo.

## Development notes

- The model code lives in `CTM/model/` and is designed to be importable as a package. The main pipeline uses relative imports to ensure `python -m CTM.model.pipeline` works from the repository root.
- Utility functions (loss/accuracy/checkpoint helpers) are in `CTM/model/utils.py`.