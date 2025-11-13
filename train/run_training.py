"""Simple training script that trains a model and visualizes loss history.

This script provides a simple way to train a SimpleTransformer model and
visualize the training progress, based on the training code from
`labs/solution-02-attention.ipynb`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_simpletransformer import SimpleTransformer
from train.dataset import TimeSeriesDataset
from train.train import train_model, visualize_attention


def main():
    """Train a SimpleTransformer model and visualize the loss history."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration (matching notebook defaults)
    input_size = 1
    seq_len = 40
    d_model = 64
    dim_feedforward = 256
    num_layers = 3
    num_samples = 1000
    batch_size = 128
    num_epochs = 15
    learning_rate = 0.0005
    dropout = 0.1

    print("=" * 60)
    print("SimpleTransformer Training Script")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Feed-forward dimension: {dim_feedforward}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Training samples: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print("=" * 60)

    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = TimeSeriesDataset(seq_len=seq_len, num_samples=num_samples, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"✅ Dataset created with {len(dataset)} samples")

    # Create model
    print("\n🤖 Initializing model...")
    model = SimpleTransformer(
        input_size=input_size,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        seq_len=seq_len,
        dropout=dropout,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model initialized with {num_params:,} parameters")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    # Define forward function to handle input reshaping and tuple return
    def forward_fn(m, x):
        # Reshape inputs: [batch, seq_len] -> [batch, seq_len, input_size]
        if x.dim() == 2:
            x = x.unsqueeze(2)
        # SimpleTransformer returns (output, attention_weights), extract just output
        output, _ = m(x)
        return output
    
    loss_history = train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        num_epochs,
        plot_loss=True,  # This will show the plot
        verbose=True,
        forward_fn=forward_fn,
    )
    print("-" * 60)

    # Final summary
    print(f"\n✅ Training completed!")
    print(f"   Initial loss: {loss_history[0]:.6f}")
    print(f"   Final loss: {loss_history[-1]:.6f}")
    print(f"   Improvement: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%")

    # Visualize attention weights
    print("\n👁️  Visualizing attention weights...")
    
    # Preprocessor for time series: add feature dimension
    def preprocess(x):
        if x.dim() == 1:
            return x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 2:
            return x.unsqueeze(2)
        return x
    
    visualize_attention(
        model,
        dataset,
        input_preprocessor=preprocess,
    )

    # Keep the plot windows open
    print("\n📈 Plots displayed. Close the plot windows to exit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()

