# Lab03: Setup a project from scratch

## Project Structure

```
project-skeleton-main/
  data/
    loader.py          # General-purpose dataset loader utilities
    mnist.py           # Example MNIST loader (original tutorial-style)
  models/              # Put model architectures/builders here
  utils/               # Put generic helpers (if any) here
  dataset/             # Downloaded datasets live here by default (e.g., dataset/mnist)
  train.py             # Training entrypoint (you customize)
  eval.py              # Evaluation entrypoint (you customize)
  requirements.txt
  README.md
```

## Data Loading

The `data/loader.py` file exposes a single function `load_data(...)` to download (if needed) and load datasets.

Supported datasets:
- mnist, fashionmnist, cifar10, cifar100 (via `torchvision.datasets`)
- imagefolder (any directory tree with `train/` and `val/` subfolders, e.g., Tiny-ImageNet)

### Quick Start (Programmatic)

```python
from data.loader import load_data

# MNIST (auto-downloads to dataset/)
train_loader, val_loader, num_classes, num_samples, train_ds = load_data(
    dataset="mnist", root="dataset", batch_size=128, num_workers=4
)

# CIFAR-10
train_loader, val_loader, num_classes, num_samples, _ = load_data(
    dataset="cifar10", root="dataset", batch_size=256
)

# Tiny-ImageNet (auto-download zip and extract, then load ImageFolder)
train_loader, val_loader, num_classes, num_samples, _ = load_data(
    dataset="imagefolder",
    root="dataset/tiny-imagenet-200",
    url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    image_size=64,
)

# Generic ImageFolder (expects dataset_root/train and dataset_root/val)
train_loader, val_loader, num_classes, num_samples, _ = load_data(
    dataset="imagefolder", root="dataset/my_images", image_size=224
)
```

### Return values and `train_ds`

`load_data(...)` returns a 5-tuple:

- `train_loader`: DataLoader for the training set
- `val_loader`: DataLoader for the validation set
- `num_classes`: number of classes
- `num_samples`: number of training samples
- `train_ds`: the underlying training Dataset object

`train_ds` lets you inspect metadata or access raw samples:

```python
print(len(train_ds))                # number of training samples
print(getattr(train_ds, 'classes', None))      # list of class names (ImageFolder/CIFAR)
print(getattr(train_ds, 'class_to_idx', None)) # mapping name -> id (ImageFolder)
img0, label0 = train_ds[0]         # a single (image_tensor, label) sample
```

### Notes

- By default, datasets are stored under `dataset/` in the project root.
- Normalization presets are applied per dataset (ImageNet, MNIST, CIFAR-10/100).

## Adding Your Own Dataset

For a custom classification dataset in folders, arrange your data as:

```
dataset/my_dataset/
  train/
    class_a/ img1.jpg ...
    class_b/ img2.jpg ...
  val/
    class_a/ ...
    class_b/ ...
```

Then call:

```python
from data.loader import load_data

train_loader, val_loader, num_classes, num_samples, _ = load_data(
    dataset="imagefolder", root="dataset/my_dataset", image_size=224
)
```

## Environment Setup (Windows, PowerShell)

### Create and activate a virtual environment
```powershell
python -m venv .venv
# or: py -3 -m venv .venv
\.venv\Scripts\Activate.ps1
# If policy error: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Upgrade pip and install dependencies
```powershell
python -m pip install --upgrade pip
# Option A (recommended): install PyTorch explicitly, then rest without deps
# CPU-only (most compatible):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# or CUDA 12.4 build:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining deps from requirements.txt (keeps your torch build)
pip install -r requirements.txt --no-deps
```

### Verify PyTorch
```powershell
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

If CUDA shows False but you installed a CUDA wheel, your NVIDIA driver is likely too old. Update it from `https://www.nvidia.com/Download/index.aspx` (Studio or Game Ready). R550+ drivers (≥ 551.61) work well with CUDA 12.4 wheels. You generally do not need the full CUDA Toolkit for PyTorch.

## Tiny-ImageNet: Train and Evaluate

### Train
```powershell
python train.py
```
This will auto-download Tiny-ImageNet (if missing), train ResNet-18 for 60 epochs with cosine LR, and save checkpoints under `checkpoints/` (`latest_*.pt`, `best_*.pt`).

Tips for CPU-only training: reduce `batch_size` (e.g., 64) and/or `epochs` (e.g., 10) via code edits if needed.

### Evaluate (validation accuracy)
```powershell
python eval.py
```
Reports top-1 and top-5 accuracy on the validation set, loading the best checkpoint if found.

### Expected accuracies (baseline)
- Training accuracy: ~80–90% top-1 by the end.
- Validation accuracy (top-1): ~45–55%.
- Validation accuracy (top-5): ~75–85%.

Numbers vary by hardware, augmentation, and hyperparameters. Stronger augmentation and regularization can push higher.

## Weights & Biases (wandb) Integration

We provide optional experiment tracking using wandb via a separate helper `utils/wandb_utils.py` and a command flag.

### Install and login
```powershell
pip install wandb
wandb login
# paste API key from https://wandb.ai/authorize
```

### Enable logging during training
```powershell
python train.py --wandb --project tiny-imagenet --run-name resnet18-baseline
```
Logged each epoch: train/val loss, train/val accuracy, learning rate. The best checkpoint is also uploaded as an artifact.

### Offline mode (no network)
```powershell
setx WANDB_MODE offline
# reopen terminal, then
python train.py --wandb
```

### Where the integration lives
- `utils/wandb_utils.py`: helpers to `init_run`, `log_metrics`, `log_checkpoint`, `finish`.
- `train.py`: optional flags `--wandb`, `--project`, `--run-name` and logging calls guarded to work even if wandb is absent.

## Troubleshooting

- ModuleNotFoundError (e.g., `requests`, `urllib3`, `idna`):
  - `pip install -r requirements.txt` (or install missing package individually)
- PyTorch wheel not found: ensure 64-bit Python 3.12 (recommended). For CUDA wheels, use the official index URLs:
  - CPU: `https://download.pytorch.org/whl/cpu`
  - CUDA 12.4: `https://download.pytorch.org/whl/cu124`
- CUDA disabled (`CUDA: False`): update NVIDIA driver; you typically don’t need the full CUDA Toolkit for PyTorch.

