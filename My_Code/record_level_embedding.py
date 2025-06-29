"""
record_level_embedder.py

This module implements an automated Record Level Embedder for tabular data
with numerical and one-hot encoded categorical features, based on the
embedding approach from the provided framework.

Features:
- Automatically detects one-hot encoded categorical blocks by grouping columns
  based on shared prefixes (splitting on last underscore).
- Supports numerical columns as individual inputs.
- Builds a Keras model with a Dense projection layer that produces
  a dense embedding vector for each record.
- Provides methods to transform input DataFrames into embedded feature arrays or DataFrames.

Usage:
    1. Instantiate RecordLevelEmbedder with your selected DataFrame and numerical columns list.
    2. Call transform_to_df() to get the embedded features as a DataFrame.
"""

from collections import defaultdict
import pandas as pd
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate


class RecordLevelEmbed:
    """
    Simple record-level embedder layer that concatenates inputs
    and applies a Dense projection to embed the full record into
    a continuous vector space.
    """

    def __init__(self, embed_dimension: int):
        self.embed_dimension = embed_dimension

    def apply(self, inputs):
        """
        Apply the embedding to the list of Keras inputs.
        Args:
            inputs (list): List of Keras Input layers or tensors.
        Returns:
            Tensor: Embedded output tensor.
        """
        print(f"[INFO] Concatenating {len(inputs)} inputs and applying Dense projection to dimension {self.embed_dimension}.")
        x = Concatenate(axis=-1)(inputs)
        x = Dense(self.embed_dimension, activation="linear")(x)
        print(f"[INFO] Projection complete. Output embedding shape: ({self.embed_dimension},)")
        return x


def detect_categorical_blocks(column_names, numerical_columns):
    """
    Automatically detect one-hot encoded categorical blocks by grouping
    columns based on prefix before the last underscore.
    Columns without underscores are treated as single-column blocks.

    Args:
        column_names (list[str]): List of all columns in the DataFrame.
        numerical_columns (list[str]): List of numerical column names to exclude.

    Returns:
        dict[str, list[str]]: Mapping of block names to list of columns in that block.
    """
    blocks = defaultdict(list)

    print("[INFO] Detecting categorical blocks...")
    for col in column_names:
        if col in numerical_columns:
            continue
        if "_" in col:
            block_name = col.rsplit("_", 1)[0]
        else:
            block_name = col
        blocks[block_name].append(col)

    print(f"[INFO] Detected {len(blocks)} categorical blocks:")
    for block, cols in blocks.items():
        print(f"   - {block}: {len(cols)} columns")

    return dict(blocks)


class RecordLevelEmbedder:
    """
    A class to build and apply a Record Level Embedding model
    for tabular data with numerical and one-hot encoded categorical features.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the selected features.
        numerical_columns (list[str]): List of numerical feature column names.
        categorical_blocks (dict[str, list[str]]): Automatically detected
            categorical blocks mapping block name to columns.
        embed_dimension (int): Output embedding dimension.
        model (tf.keras.Model): The underlying Keras model for embedding.
    """

    def __init__(self, selected_df: pd.DataFrame, numerical_columns: list, embed_dimension: int = 64):
        """
        Initialize the embedder.

        Args:
            selected_df (pd.DataFrame): DataFrame of selected features (numerical + one-hot).
            numerical_columns (list[str]): List of numerical columns.
            embed_dimension (int): Dimension of output embedding vector.
        """
        print("[INFO] Initializing RecordLevelEmbedder...")
        self.df = selected_df
        self.numerical_columns = numerical_columns
        self.embed_dimension = embed_dimension
        self.categorical_blocks = detect_categorical_blocks(
            column_names=self.df.columns.tolist(),
            numerical_columns=self.numerical_columns
        )
        self.model = None
        print(f"[INFO] Embedder initialized with embedding dimension {self.embed_dimension}.")

    def build_model(self):
        """
        Build the Keras model to perform record-level embedding.
        """
        print("[INFO] Building Keras embedding model...")
        inputs = []

        # Numerical inputs
        for col in self.numerical_columns:
            print(f"   - Adding numerical input: {col}")
            inp = Input(shape=(1,), name=col)
            inputs.append(inp)

        # Categorical blocks
        for block_name, block_cols in self.categorical_blocks.items():
            print(f"   - Adding categorical block input: {block_name} ({len(block_cols)} columns)")
            inp = Input(shape=(len(block_cols),), name=block_name)
            inputs.append(inp)

        # Apply embedding
        embedder = RecordLevelEmbed(embed_dimension=self.embed_dimension)
        embedded = embedder.apply(inputs)

        self.model = Model(inputs=inputs, outputs=embedded)
        print("[INFO] Model build complete.")

    def transform(self, df: pd.DataFrame = None) -> np.ndarray:
        """
        Transform the given DataFrame into embedded feature vectors.

        Args:
            df (pd.DataFrame, optional): DataFrame to transform.
                If None, uses the DataFrame passed during initialization.

        Returns:
            np.ndarray: Embedded feature vectors of shape (num_samples, embed_dimension).
        """
        if df is None:
            df = self.df

        if self.model is None:
            self.build_model()

        print(f"[INFO] Preparing inputs for embedding. Number of rows: {len(df)}")
        X_inputs = {}

        # Numerical inputs
        for col in self.numerical_columns:
            X_inputs[col] = df[col].values.reshape(-1, 1)
            print(f"   - Prepared numerical input: {col}")

        # Categorical inputs
        for block_name, block_cols in self.categorical_blocks.items():
            X_inputs[block_name] = df[block_cols].values
            print(f"   - Prepared categorical block input: {block_name}")

        print("[INFO] Performing embedding inference...")
        embeddings = self.model.predict(X_inputs, verbose=1)
        print(f"[INFO] Embedding complete. Output shape: {embeddings.shape}")
        return embeddings

    def transform_to_df(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Transform given DataFrame and return the embeddings as a pandas DataFrame.

        Args:
            df (pd.DataFrame, optional): DataFrame to transform.
                If None, uses the DataFrame passed during initialization.

        Returns:
            pd.DataFrame: Embedded features as a DataFrame with columns embed_0, embed_1, ...
        """
        embeddings = self.transform(df)
        embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
        index = df.index if df is not None else self.df.index
        print("[INFO] Embedding DataFrame ready.")
        return pd.DataFrame(embeddings, columns=embed_cols, index=index)
