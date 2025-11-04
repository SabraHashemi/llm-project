from data.loader import load_data
import os
import torch
from torch import nn
from models.resnet import create_resnet18_classifier


def main() -> None:
    # Load validation data
    _, val_loader, num_classes, _, _ = load_data(
        dataset="imagefolder",
        root="dataset/tiny-imagenet-200",
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        image_size=64,
        batch_size=256,
        num_workers=4,
        download=True,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = create_resnet18_classifier(num_classes=num_classes, pretrained=False)

    # Find best checkpoint
    ckpt_dir = "checkpoints"
    best_weights = os.path.join(ckpt_dir, "best_resnet18_tinyimagenet.pt")
    latest_state = os.path.join(ckpt_dir, "latest_resnet18_tinyimagenet.pt")

    if os.path.exists(best_weights):
        state_dict = torch.load(best_weights, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded best weights: {best_weights}")
    elif os.path.exists(latest_state):
        bundle = torch.load(latest_state, map_location="cpu")
        model.load_state_dict(bundle["model_state"]) if isinstance(bundle, dict) and "model_state" in bundle else model.load_state_dict(bundle)
        print(f"Loaded latest state: {latest_state}")
    else:
        print("No checkpoint found. Evaluating randomly initialized model.")

    model.to(device)
    model.eval()

    total = 0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            # top-1
            pred1 = outputs.argmax(dim=1)
            correct_top1 += (pred1 == targets).sum().item()
            # top-5
            top5 = outputs.topk(5, dim=1).indices
            correct_top5 += top5.eq(targets.view(-1, 1)).any(dim=1).sum().item()
            total += targets.size(0)

    top1_acc = correct_top1 / max(1, total)
    top5_acc = correct_top5 / max(1, total)
    print(f"Validation accuracy: top-1={top1_acc*100:.2f}% | top-5={top5_acc*100:.2f}% (N={total})")


if __name__ == "__main__":
    main()

