"""
basic_transformer.py

This module implements a simple Transformer Encoder stack for tabular or embedded data.

- Uses only encoder blocks (no decoder)
- Supports stacking multiple layers with multi-head self-attention
- Optional verbosity for debugging

Example:
    from basic_transformer import BasicTransformer

    transformer = BasicTransformer(n_layers=2, internal_size=128, n_heads=4)
    output = transformer.apply(input_tensor)
"""

import warnings
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, MultiHeadAttention, Dropout, LayerNormalization, Conv1D


class TransformerEncoderBlock(layers.Layer):
    """
    A single Transformer Encoder block:
    - Multi-Head Self-Attention
    - Residual connection + LayerNorm
    - Feed-forward (Dense or Conv1D)
    - Residual connection + LayerNorm
    """

    def __init__(self, input_dimension: int, inner_dimension: int, num_heads: int,
                 dropout_rate: float = 0.1, use_conv: bool = False, prefix: str = "", verbose: bool = False):
        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn("Typically inner_dimension should be >= input_dimension!")

        self.verbose = verbose

        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension,
                                            name=f"{prefix}multi_head_attention")
        self.attention_dropout = Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = LayerNormalization(epsilon=1e-6, name=f"{prefix}attention_layer_norm")

        if use_conv:
            self.feed_forward_0 = Conv1D(filters=inner_dimension, kernel_size=1, activation="relu",
                                         name=f"{prefix}feed_forward_0")
            self.feed_forward_1 = Conv1D(filters=input_dimension, kernel_size=1, activation="relu",
                                         name=f"{prefix}feed_forward_1")
        else:
            self.feed_forward_0 = Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")
            self.feed_forward_1 = Dense(input_dimension, activation="relu", name=f"{prefix}feed_forward_1")

        self.feed_forward_dropout = Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = LayerNormalization(epsilon=1e-6, name=f"{prefix}feed_forward_layer_norm")

    def call(self, inputs, training=None):
        if self.verbose:
            print(f"[EncoderBlock] Input shape: {inputs.shape}")

        # Multi-head self-attention
        attention_output = self.attention(inputs, inputs)
        attention_output = self.attention_dropout(attention_output, training=training)
        x = self.attention_layer_norm(inputs + attention_output)

        # Feed-forward network
        ffn_output = self.feed_forward_0(x)
        ffn_output = self.feed_forward_1(ffn_output)
        ffn_output = self.feed_forward_dropout(ffn_output, training=training)

        output = self.feed_forward_layer_norm(x + ffn_output)

        if self.verbose:
            print(f"[EncoderBlock] Output shape: {output.shape}")

        return output


class BasicTransformer:
    """
    Simple Transformer Encoder stack for tabular/embedded data.
    """

    def __init__(self, n_layers: int, internal_size: int, n_heads: int,
                 use_conv: bool = False, dropout_rate: float = 0.1, verbose: bool = False):
        """
        Args:
            n_layers (int): Number of encoder layers to stack.
            internal_size (int): Size of the feed-forward hidden layer.
            n_heads (int): Number of attention heads.
            use_conv (bool): Whether to use Conv1D in feed-forward (optional).
            dropout_rate (float): Dropout rate.
            verbose (bool): If True, prints progress.
        """
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.use_conv = use_conv
        self.dropout_rate = dropout_rate
        self.verbose = verbose

    def apply(self, X, prefix: str = "", training=False):
        """
        Apply the stacked Transformer Encoder layers.

        Args:
            X (tf.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim)
                           For tabular data, sequence_length can be 1.
            prefix (str): Optional name prefix for layers.
            training (bool): Training mode.

        Returns:
            tf.Tensor: Output tensor with same shape as input, but transformed.
        """
        output = X
        real_size = X.shape[-1]

        for layer_i in range(self.n_layers):
            if self.verbose:
                print(f"[BasicTransformer] Applying Encoder Block {layer_i+1}/{self.n_layers}")
            output = TransformerEncoderBlock(
                input_dimension=real_size,
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                use_conv=self.use_conv,
                prefix=f"{prefix}block_{layer_i}_",
                verbose=self.verbose
            )(output, training=training)

        if self.verbose:
            print(f"[BasicTransformer] Final output shape: {output.shape}")

        return output
