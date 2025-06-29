"""
last_token_classification_head.py

A simple classification head that selects the last token
from a sequence output (e.g., from a Transformer encoder)
and outputs it for downstream classification.

Typical use case:
    (batch_size, sequence_length, embed_dim) → (batch_size, embed_dim)

Usage:
    head = LastTokenClassificationHead()
    pooled = head.apply(transformer_output)
"""

import tensorflow as tf
from tensorflow.keras.layers import Lambda

# Optional: base class if you want consistent interface
class BaseClassificationHead:
    def apply(self, X, prefix: str = None):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "BaseClassificationHead"

    @property
    def parameters(self) -> dict:
        return {}

class LastTokenClassificationHead(BaseClassificationHead):
    def __init__(self):
        super().__init__()

    def apply(self, X, prefix: str = None):
        """
        Selects the last token embedding from the input sequence tensor.
        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            prefix (str, optional): Prefix for layer name.
        Returns:
            Tensor: Output tensor of shape (batch_size, embed_dim)
        """
        if prefix is None:
            prefix = ""

        return Lambda(
            lambda x: x[:, -1, :],
            name=f"{prefix}slice_last"
        )(X)

    @property
    def name(self) -> str:
        return "LastTokenClassificationHead"

    @property
    def parameters(self) -> dict:
        return {}
