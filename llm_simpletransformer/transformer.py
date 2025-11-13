from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn

from llm_attention import TransformerEncoderBlock


class SimpleTransformer(nn.Module):
    """Simple Transformer model for sequence-to-value tasks (e.g., time series).

    This implementation mirrors the Transformer model from
    `labs/solution-02-attention.ipynb` and makes it reusable across the project.
    It's designed for tasks where you want to process a sequence and output a
    single value (like time series regression).

    The model consists of:
    1. **Embedding Layer**: Projects input features to `d_model` dimensions
    2. **Positional Encoding** (optional): Adds positional information if provided
    3. **Encoder Layers**: Stack of `TransformerEncoderBlock` modules
    4. **Output Layer**: Linear projection to output size (default: 1 for regression)

    Parameters
    ----------
    input_size:
        Number of input features per time step.
    d_model:
        Dimensionality of the model embeddings.
    dim_feedforward:
        Dimensionality of the intermediate feed-forward representation in each
        encoder block.
    num_layers:
        Number of stacked encoder blocks.
    seq_len:
        Maximum sequence length expected by the model.
    positional_encoding:
        Optional callable that takes `(d_model, max_len)` and returns a
        positional encoding module. If provided, relative positional encodings
        in the attention layers are disabled.
    output_size:
        Size of the output (default: 1 for regression tasks).
    dropout:
        Dropout probability applied in encoder blocks.
    max_len:
        Maximum sequence length for positional encoding (if provided).
    use_mask:
        Whether the model should accept attention masks in forward passes.
    device, dtype:
        Optional factory kwargs for submodules.

    Examples
    --------
    >>> from llm_simpletransformer import SimpleTransformer
    >>> model = SimpleTransformer(
    ...     input_size=1,
    ...     d_model=64,
    ...     dim_feedforward=256,
    ...     num_layers=3,
    ...     seq_len=40
    ... )
    >>> x = torch.randn(32, 40, 1)  # [batch, seq_len, input_size]
    >>> output, attention_weights = model(x)
    >>> print(output.shape)  # [32, 1]
    >>> print(len(attention_weights))  # 3 (one per layer)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        dim_feedforward: int,
        num_layers: int,
        seq_len: int,
        *,
        positional_encoding: Optional[Callable[[int, int], nn.Module]] = None,
        output_size: int = 1,
        dropout: float = 0.1,
        max_len: int = 5000,
        use_mask: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.positional_encoding = positional_encoding
        self.use_mask = use_mask

        # Embedding layer: projects input features to d_model dimensions
        self.embedding = nn.Linear(input_size, d_model, **factory_kwargs)

        # Positional encoding (optional)
        if self.positional_encoding is not None:
            self.pos_encoder = self.positional_encoding(d_model, max_len)
            other_pos_encodings_present = True
        else:
            self.register_parameter("pos_encoder", None)
            other_pos_encodings_present = False

        # Stack of encoder blocks
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model,
                    dim_feedforward,
                    seq_len,
                    other_pos_encodings_present,
                    dropout=dropout,
                    use_mask=use_mask,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Output layer: projects from d_model to output_size
        self.fc_out = nn.Linear(d_model, output_size, **factory_kwargs)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Run the transformer forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``[batch_size, seq_len, input_size]``.
        mask:
            Optional attention mask broadcastable to attention scores. Only used
            if the model was initialized with ``use_mask=True``.

        Returns
        -------
        output:
            Tensor of shape ``[batch_size, output_size]``. For regression tasks
            with ``output_size=1``, this is the predicted value based on the
            last time step of the sequence.
        all_attention_weights:
            List of attention weight tensors, one per encoder layer. Each tensor
            has shape ``[batch, seq_len, seq_len]`` showing how each position
            attends to all other positions.
        """

        # Embed input: [batch, seq_len, input_size] -> [batch, seq_len, d_model]
        x = self.embedding(x)

        # Add positional encoding if provided
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # Collect attention weights from all layers
        all_attention_weights: List[Tensor] = []

        # Pass through each encoder layer
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, mask=mask)
            all_attention_weights.append(attention_weights)

        # Output: take the last time step and project to output size
        # [batch, seq_len, d_model] -> [batch, d_model] -> [batch, output_size]
        output = self.fc_out(x[:, -1])

        return output, all_attention_weights

