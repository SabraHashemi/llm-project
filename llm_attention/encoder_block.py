from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .attention_head import AttentionHead


class TransformerEncoderBlock(nn.Module):
    """Single encoder block composed of attention + feed-forward network.

    This mirrors the implementation used in `labs/solution-02-attention.ipynb`
    and can be dropped into other models. It internally relies on the modular
    :class:`AttentionHead` and exposes the same option for skipping relative
    positional encodings when another mechanism is applied upstream.

    Parameters
    ----------
    d_model:
        Dimensionality of the model embeddings.
    dim_feedforward:
        Dimensionality of the intermediate feed-forward representation.
    seq_len:
        Maximum sequence length processed by the attention head.
    other_positional_encodings_present:
        Whether positional information is already injected elsewhere. Passed to
        the attention head to control relative encodings.
    dropout:
        Dropout probability applied after attention and feed-forward sublayers.
    use_mask:
        Forward calls may include an attention mask when set to ``True``.
    device, dtype:
        Optional factory kwargs for submodules.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        seq_len: int,
        other_positional_encodings_present: bool,
        *,
        dropout: float = 0.1,
        use_mask: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.attention = AttentionHead(
            d_model,
            seq_len,
            other_positional_encodings_present,
            use_mask=use_mask,
            device=device,
            dtype=dtype,
        )
        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model, **factory_kwargs),
        )

        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the encoder block.

        Parameters
        ----------
        x:
            Input tensor of shape ``[batch, seq_len, d_model]``.
        mask:
            Optional attention mask broadcastable to the attention scores.

        Returns
        -------
        output:
            Tensor of shape ``[batch, seq_len, d_model]`` after the encoder
            block.
        attention_weights:
            Attention probabilities from the internal attention head.
        """

        attn_out, attn_weights = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        output = self.norm2(x + self.dropout(ffn_out))

        return output, attn_weights
