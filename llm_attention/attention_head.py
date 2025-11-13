from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .scaled_dot_product import ScaledDotProductAttention


class AttentionHead(nn.Module):
    """Single attention head built on top of scaled dot-product attention.

    The implementation mirrors the notebook used in ``labs/solution-02-attention``
    while exposing a reusable, modular component. Queries, keys and values are
    first projected through learnable linear layers before the
    :class:`ScaledDotProductAttention` module is applied. The output is then
    passed through a final linear projection.

    Parameters
    ----------
    d_model:
        Dimensionality of the model embeddings. The query, key and value
        projections map from ``d_model`` to ``d_model``.
    seq_len:
        Maximum sequence length handled by this attention head. It is forwarded
        to :class:`ScaledDotProductAttention`.
    other_positional_encodings_present:
        Whether an external positional encoding mechanism is already used. When
        ``False`` the underlying attention module learns a relative positional
        encoding, keeping parity with the notebook implementation.
    use_mask:
        If ``True`` the :meth:`forward` method may accept an additional mask
        tensor to block out specific key positions.
    device, dtype:
        Optional factory keyword arguments passed to the internal linear
        layers.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        other_positional_encodings_present: bool,
        *,
        use_mask: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.query_layer = nn.Linear(d_model, d_model, **factory_kwargs)
        self.key_layer = nn.Linear(d_model, d_model, **factory_kwargs)
        self.value_layer = nn.Linear(d_model, d_model, **factory_kwargs)

        self.scaled_dot_attention = ScaledDotProductAttention(
            seq_len,
            d_model,
            other_positional_encodings_present,
            use_mask=use_mask,
            device=device,
            dtype=dtype,
        )

        self.fc_out = nn.Linear(d_model, d_model, **factory_kwargs)
        self.use_mask = use_mask

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the attention head forward pass.

        Parameters
        ----------
        query, key, value:
            Tensors of shape ``[batch, seq_len, d_model]`` (or compatible
            shapes).
        mask:
            Optional attention mask broadcastable to the scores computed inside
            :class:`ScaledDotProductAttention`.

        Returns
        -------
        output:
            Tensor of shape ``[batch, seq_len, d_model]`` after the output
            projection.
        attention_weights:
            Attention probability tensor returned by the underlying scaled
            dot-product attention.
        """

        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)

        attn_output, attn_weights = self.scaled_dot_attention(Q, K, V, mask=mask)

        if attn_output.dim() == 3 and attn_output.size(1) == 1:
            attn_output = attn_output.squeeze(1)

        output = self.fc_out(attn_output)
        return output, attn_weights
