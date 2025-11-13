from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    """Compute scaled dot-product attention with optional relative encodings.

    This module mirrors the implementation from the `solution-02-attention.ipynb`
    notebook and makes it reusable across the project. It supports the same
    behaviour as the notebook cell while exposing additional conveniences like
    an optional attention mask.

    Parameters
    ----------
    seq_len:
        Maximum sequence length expected by the module. When relative positional
        encodings are enabled the learnable parameter has shape
        ``[1, seq_len, seq_len]`` and is sliced to match shorter sequences at
        runtime.
    d_k:
        Dimensionality of the key/query vectors. Used for the :math:`1/\sqrt{d_k}`
        scaling factor from the original Transformer paper.
    other_positional_encodings_present:
        If ``True`` the module skips adding learnable relative positional
        encodings because another component is already handling positional
        information.
    use_mask:
        Whether to expect an attention mask in :meth:`forward`. The mask is
        optional even when ``use_mask`` is ``True``; the flag simply documents
        intended usage and keeps parity with the notebook implementation where a
        mask was not required by default.
    device:
        Optional device on which the learnable parameters should be initialised.
    dtype:
        Optional data type for the learnable parameters.
    """

    def __init__(
        self,
        seq_len: int,
        d_k: int,
        other_positional_encodings_present: bool,
        *,
        use_mask: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.d_k = float(d_k)
        self.other_positional_encodings_present = other_positional_encodings_present
        self.use_mask = use_mask

        if not other_positional_encodings_present:
            # Learnable relative positional encoding replicated from the notebook
            # implementation.  The parameter is sliced in forward() to support
            # shorter sequence lengths at inference time.
            factory_kwargs = {"device": device, "dtype": dtype}
            self.rel_layer = nn.Parameter(
                torch.randn(1, seq_len, seq_len, requires_grad=True, **factory_kwargs)
            )
        else:
            self.register_parameter("rel_layer", None)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Execute the attention operation.

        Parameters
        ----------
        query, key, value:
            Tensors of shape ``[batch, seq_len, d_model]`` (or compatible
            shapes) representing the standard Transformer triplet.
        mask:
            Optional boolean mask where ``0`` (or ``False``) locations are
            ignored. The mask is broadcast against the attention score tensor.

        Returns
        -------
        output:
            Weighted value tensor after applying attention.
        attention_weights:
            Softmax-normalised weights indicating the contribution of each value
            vector.
        """

        scale = 1.0 / math.sqrt(self.d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if self.rel_layer is not None:
            rel = self.rel_layer[..., : scores.size(-2), : scores.size(-1)]
            scores = scores + rel

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
