from typing import Optional, Tuple
import os, requests, zipfile
from io import BytesIO
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


def _tv_datasets(name: str, root: str, download: bool, train_tf, val_tf) -> Tuple[Dataset, Dataset, int]:
    name = name.lower()
    if name == "mnist":
        return datasets.MNIST(root, True, download, train_tf), datasets.MNIST(root, False, download, val_tf), 10
    if name == "fashionmnist":
        return datasets.FashionMNIST(root, True, download, train_tf), datasets.FashionMNIST(root, False, download, val_tf), 10
    if name == "cifar10":
        return datasets.CIFAR10(root, True, download, train_tf), datasets.CIFAR10(root, False, download, val_tf), 10
    if name == "cifar100":
        return datasets.CIFAR100(root, True, download, train_tf), datasets.CIFAR100(root, False, download, val_tf), 100
    raise ValueError("unsupported torchvision dataset")


def _transforms(name: str, image_size: Optional[int]):
    name = name.lower()
    if name in {"mnist", "fashionmnist"}:
        norm = transforms.Normalize((0.1307,), (0.3081,))
        return transforms.Compose([transforms.ToTensor(), norm]), transforms.Compose([transforms.ToTensor(), norm])
    if name in {"cifar10", "cifar100"}:
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        return (
            transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm]),
            transforms.Compose([transforms.ToTensor(), norm]),
        )
    # imagefolder
    tf = [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    if image_size:
        tf.insert(0, transforms.Resize((image_size, image_size)))
    c = transforms.Compose(tf)
    return c, c


def _maybe_download_zip(url: Optional[str], dest_dir: str) -> None:
    if not url:
        return
    os.makedirs(dest_dir, exist_ok=True)
    r = requests.get(url)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        zf.extractall(dest_dir)


def load_data(
    dataset: str,
    root: str = "dataset",
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    image_size: Optional[int] = None,
    url: Optional[str] = None,  # only for imagefolder zips
):
    """
    Download (if needed) and load a dataset, returning loaders and metadata.

    Returns:
        train_loader:  PyTorch DataLoader for training
        val_loader:    PyTorch DataLoader for validation
        num_classes:   int, number of classes
        num_samples:   int, number of training samples
        train_ds:      Dataset object (torchvision dataset or ImageFolder)
    """
    name = dataset.lower()
    train_tf, val_tf = _transforms(name, image_size)
    if name in {"mnist", "fashionmnist", "cifar10", "cifar100"}:
        train_ds, val_ds, num_classes = _tv_datasets(name, root, download, train_tf, val_tf)
    else:
        # imagefolder (optionally auto-download zip)
        _maybe_download_zip(url, os.path.dirname(root) or ".")
        training_dataset = ImageFolder(os.path.join(root, "train"), transform=train_tf)
        val_ds = ImageFolder(os.path.join(root, "val"), transform=val_tf)
        num_classes = len(training_dataset.classes)

    train_loader = DataLoader(training_dataset if name not in {"mnist", "fashionmnist", "cifar10", "cifar100"} else train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    if name in {"mnist", "fashionmnist", "cifar10", "cifar100"}:
        training_dataset = train_ds
    train_ds = training_dataset
    return train_loader, val_loader, num_classes, len(train_ds), train_ds


