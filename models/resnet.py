from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tvm


def create_resnet18_classifier(num_classes: int, pretrained: bool = False, dropout_p: Optional[float] = 0.2) -> nn.Module:
    """
    Create a ResNet-18 backbone adapted for Tiny-ImageNet classification (64x64 RGB).

    Args:
        num_classes: Number of target classes.
        pretrained: If True, load ImageNet-1k pretrained weights.
        dropout_p: Optional dropout probability before the final classifier.

    Returns:
        nn.Module ready to train/evaluate.
    """
    # ResNet-18 works well for 64x64; keep the default stem. Optionally use pretrained weights.
    if pretrained:
        try:
            model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            model = tvm.resnet18(weights=None)
    else:
        model = tvm.resnet18(weights=None)

    in_features = model.fc.in_features
    head: list[nn.Module] = []
    if dropout_p is not None and dropout_p > 0:
        head.append(nn.Dropout(p=dropout_p))
    head.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*head)
    return model


