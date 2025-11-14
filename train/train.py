"""General-purpose training utilities for PyTorch models.

This module provides reusable training functions that work with any PyTorch model
and dataset. It's based on the training code from `labs/solution-02-attention.ipynb`
but made generic to work with any model/dataset combination.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    *,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    plot_loss: bool = True,
    forward_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
) -> list[float]:
    """Train a model for the specified number of epochs.

    This is a general-purpose training function that works with any PyTorch model
    and dataset. The user is responsible for providing properly formatted data
    from their DataLoader.

    Parameters
    ----------
    model:
        The model to train.
    dataloader:
        DataLoader providing training batches. Each batch should be a tuple of
        (inputs, targets) where inputs and targets are tensors ready for the model.
    criterion:
        Loss function.
    optimizer:
        Optimizer for updating model parameters.
    num_epochs:
        Number of training epochs.
    device:
        Device to run training on (default: auto-detect).
    verbose:
        Whether to print training progress.
    plot_loss:
        Whether to plot loss history after training.
    forward_fn:
        Optional custom forward function. If provided, it will be called as
        `forward_fn(model, inputs)` instead of `model(inputs)`. This is useful
        for models that return tuples or have special forward signatures.
        The function should return the model output(s).

    Returns
    -------
    loss_history:
        List of average loss values per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    loss_history: list[float] = []

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch in dataloader:
            # Assume batch is (inputs, targets) tuple
            # User is responsible for proper formatting
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, target = batch
                inputs = inputs.to(device)
                target = target.to(device)
            else:
                # If batch is a single tensor or dict, user should handle it
                inputs = batch.to(device) if isinstance(batch, torch.Tensor) else batch
                target = None  # User must handle this case

            optimizer.zero_grad()

            # Use custom forward function if provided, otherwise standard forward
            if forward_fn is not None:
                outputs = forward_fn(model, inputs)
            else:
                outputs = model(inputs)

            # Handle models that return tuples (e.g., (output, attention_weights))
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Compute loss
            if target is not None:
                # Handle different output shapes
                if outputs.dim() > target.dim():
                    outputs = outputs.squeeze()
                loss = criterion(outputs, target)
            else:
                raise ValueError(
                    "Could not determine target from batch. "
                    "Please ensure your DataLoader returns (inputs, targets) tuples, "
                    "or provide a custom forward_fn."
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    if plot_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return loss_history


def visualize_attention(
    model: nn.Module,
    dataset: Dataset,
    *,
    device: Optional[torch.device] = None,
    sample_idx: Optional[int] = None,
    input_preprocessor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    """Visualize attention weights for a sample from the dataset.

    This is a general-purpose visualization function that works with any model
    that returns attention weights. The user must provide appropriate preprocessing
    for their specific dataset format.

    Parameters
    ----------
    model:
        Trained model to visualize. Should return (output, attention_weights) tuple.
    dataset:
        Dataset to sample from. Should return (inputs, target) tuple.
    device:
        Device to run inference on (default: auto-detect).
    sample_idx:
        Index of sample to visualize (default: find first suitable sample).
    input_preprocessor:
        Optional function to preprocess inputs before passing to model.
        Should take a tensor and return a tensor ready for the model.
        If None, inputs are used as-is (after moving to device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Get a sample from the dataset
    if sample_idx is not None:
        inputs, target = dataset[sample_idx]
    else:
        # Find first suitable sample
        inputs, target = None, None
        for i in range(len(dataset)):
            inputs, target = dataset[i]
            # For time series, skip zero targets; for other datasets, use first sample
            if not (hasattr(target, "item") and target.item() == 0):
                break

    if inputs is None or target is None:
        print("No suitable sample found for visualization.")
        return

    # Preprocess inputs if function provided
    if input_preprocessor is not None:
        inputs = input_preprocessor(inputs)
    else:
        # Default: add batch dimension
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
    
    # Move inputs to device after preprocessing
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        # Handle models that return tuples
        if isinstance(outputs, tuple):
            outputs, attention_weights = outputs
        else:
            print("Model does not return attention weights. Cannot visualize.")
            return

    # Plot input (assume 1D sequence for visualization)
    if inputs.dim() == 3:  # [batch, seq, features]
        inputs_np = inputs[0, :, 0].detach().cpu().numpy()
    elif inputs.dim() == 2:  # [batch, seq] or [seq, features]
        inputs_np = inputs[0].detach().cpu().numpy() if inputs.size(0) > 1 else inputs.squeeze().detach().cpu().numpy()
    else:
        inputs_np = inputs.squeeze().detach().cpu().numpy()

    plt.plot(inputs_np, label="Input Series", linewidth=2)

    # Use attention from the last layer
    if isinstance(attention_weights, (list, tuple)):
        last_layer_attn = attention_weights[-1][0].detach().cpu().numpy()
    else:
        last_layer_attn = attention_weights[0].detach().cpu().numpy()

    # Normalize attention weights for better visibility
    attn_min = last_layer_attn.min()
    attn_max = last_layer_attn.max()
    if attn_max > attn_min:
        normalized_attn = (last_layer_attn - attn_min) / (attn_max - attn_min)
        normalized_attn_mean = normalized_attn.mean(axis=0)
        plt.plot(
            normalized_attn_mean,
            label="Normalized Attention Weights",
            linestyle="--",
            linewidth=2,
        )

    # Plot true and predicted targets if available
    if hasattr(target, "item"):
        target_val = target.item()
        output_val = outputs[0].item() if outputs.numel() == 1 else outputs.squeeze()[0].item()
        seq_len = len(inputs_np)
        plt.scatter(seq_len, target_val, color="r", s=100, label="True Target", zorder=5)
        plt.scatter(seq_len, output_val, color="g", s=100, label="Predicted Target", zorder=5)
        plt.title(
            f"Attention Visualization\n"
            f"True Target: {target_val:.2f}, "
            f"Predicted: {output_val:.2f}"
        )
    else:
        plt.title("Attention Visualization")

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Example main training script for SimpleTransformer on time series data.
    
    This is an example implementation showing how to use the generic training
    functions with a specific dataset. For other use cases, create your own
    main function or use train_model() directly.
    """
    # Import here to avoid circular dependencies and keep main() optional
    from llm_simpletransformer import SimpleTransformer
    from train.dataset import TimeSeriesDataset

    parser = argparse.ArgumentParser(
        description="Train a SimpleTransformer model on synthetic time series data"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1,
        help="Number of input features per time step (default: 1)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=40,
        help="Length of input sequences (default: 40)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Model embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=256,
        help="Feed-forward network dimension (default: 256)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of encoder layers (default: 3)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of training samples (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate (default: 0.0005)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable loss plotting",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize attention weights after training",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save the trained model (optional)",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create dataset
    print(f"Creating dataset with {args.num_samples} samples...")
    dataset = TimeSeriesDataset(
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Create model
    print("Initializing model...")
    model = SimpleTransformer(
        input_size=args.input_size,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        dropout=args.dropout,
    )

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    # Train model
    # SimpleTransformer returns (output, attention_weights) tuple, so we extract just the output
    loss_history = train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        args.num_epochs,
        plot_loss=not args.no_plot,
        forward_fn=lambda m, x: m(x)[0],  # Extract output from (output, attention) tuple
    )

    print("-" * 60)
    print(f"Training completed!")
    print(f"Final loss: {loss_history[-1]:.6f}")

    # Save model if requested
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    # Visualize attention if requested
    if args.visualize:
        print("\nVisualizing attention weights...")
        # Preprocessor for time series: add feature dimension
        def preprocess(x):
            if x.dim() == 1:
                return x.unsqueeze(0).unsqueeze(2)
            return x
        visualize_attention(
            model,
            dataset,
            input_preprocessor=preprocess,
        )


if __name__ == "__main__":
    main()

