"""Time series dataset for training SimpleTransformer models."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Synthetic time series dataset with jumps at regular intervals.

    This dataset generates synthetic time series data with pseudorandom spikes
    introduced at specific points. Each time series begins as a flat line and
    has a jump introduced at a random point within the first few time steps.
    Additional jumps occur at regular intervals, alternating between small and
    large jumps.

    This implementation mirrors the dataset from `labs/solution-02-attention.ipynb`.

    Parameters
    ----------
    seq_len:
        Length of the input sequence (number of time steps).
    num_samples:
        Number of time series samples to generate.
    initial_jump:
        Magnitude of the initial jump (default: 5).
    jump_interval:
        Interval between jumps (default: 3).
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        seq_len: int,
        num_samples: int,
        *,
        initial_jump: float = 5.0,
        jump_interval: int = 3,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        self.seq_len = seq_len
        self.data: list[np.ndarray] = []

        for _ in range(num_samples):
            time_series = np.zeros(seq_len + 1)

            # Introduce the first jump at a random point between 0 and jump_interval
            first_jump = np.random.randint(0, jump_interval)
            time_series[first_jump] += np.random.rand() + initial_jump

            # Continue to introduce jumps every jump_interval timesteps after the first jump
            for i in range(first_jump + jump_interval, seq_len + 1, jump_interval):
                # Alternate between 0 and initial_jump
                jump = 0 if i % 2 == 1 else initial_jump
                time_series[i] += np.random.rand() + jump

            self.data.append(time_series)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single time series sample.

        Returns
        -------
        input_sequence:
            Tensor of shape ``[seq_len]`` containing the input time series.
        target:
            Scalar tensor containing the target value (next time step).
        """
        series = self.data[idx]
        return (
            torch.tensor(series[:-1], dtype=torch.float32),
            torch.tensor(series[-1], dtype=torch.float32),
        )

