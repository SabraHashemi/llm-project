from data.loader import load_data
import os
import math
import time
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.resnet import create_resnet18_classifier
import argparse

try:
    from utils import wandb_utils
except Exception:
    wandb_utils = None  # Optional


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)


def run_epoch(model: nn.Module, loader, criterion, optimizer=None, device: str = "cuda") -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, targets)
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, targets) * batch_size
        total += batch_size

    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_acc / max(1, total)
    return epoch_loss, epoch_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--project", type=str, default="tiny-imagenet", help="wandb project name")
    parser.add_argument("--run-name", type=str, default=None, help="optional run name")
    args = parser.parse_args()
    # Data: Tiny-ImageNet with auto-download
    train_loader, val_loader, num_classes, num_samples, train_ds = load_data(
        dataset="imagefolder",
        root="dataset/tiny-imagenet-200",
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        image_size=64,
        batch_size=256,
        num_workers=4,
        download=True,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)

    # Model
    model = create_resnet18_classifier(num_classes=num_classes, pretrained=False, dropout_p=0.2)
    model.to(device)

    # Optimizer/Scheduler/Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    epochs = 60
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_path = os.path.join("checkpoints", "best_resnet18_tinyimagenet.pt")

    # Optional wandb
    if args.wandb and wandb_utils is not None:
        wandb_utils.init_run(
            project=args.project,
            name=args.run_name,
            config={
                "model": "resnet18",
                "epochs": 60,
                "batch_size": 256,
                "lr": 0.1,
                "weight_decay": 5e-4,
                "optimizer": "sgd+nesterov",
                "scheduler": "cosine",
                "image_size": 64,
            },
        )

    print(f"Starting training for {epochs} epochs on device={device}. Classes={num_classes}, Train samples={num_samples}")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        scheduler.step()

        # Save latest
        latest_path = os.path.join("checkpoints", "latest_resnet18_tinyimagenet.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_acc": val_acc,
        }, latest_path)

        # Track best
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            if args.wandb and wandb_utils is not None:
                wandb_utils.log_checkpoint(best_path)

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:03d}/{epochs} | LR {lr:.5f} | "
            f"Train: loss {train_loss:.4f}, acc {train_acc*100:.2f}% | "
            f"Val: loss {val_loss:.4f}, acc {val_acc*100:.2f}% | "
            f"Best Val Acc: {best_val_acc*100:.2f}%"
        )
        if args.wandb and wandb_utils is not None:
            wandb_utils.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": lr,
            }, step=epoch)

    elapsed = time.time() - start_time
    print(f"Done. Best Val Acc: {best_val_acc*100:.2f}%. Training time: {elapsed/60:.1f} min. Best model -> {best_path}")
    if args.wandb and wandb_utils is not None:
        wandb_utils.finish()


if __name__ == "__main__":
    main()

