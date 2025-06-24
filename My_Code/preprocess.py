import numpy as np
import pandas as pd

class Preprocess:
    def __init__(self, clip_numerical_values:bool=False):
        self.min_range = {}
        self.encoded_levels = {}
        self.clip_numerical_values:bool = clip_numerical_values


    def __fit_numerical(self, column_name: str, values: np.array):

        v0 = np.min(values)
        v1 = np.max(values)
        r = v1 - v0

        self.min_range[column_name] = (v0, r)

    def __transform_numerical(self, column_name: str, values: np.array):
        col_min, col_range = self.min_range[column_name]

        if col_range == 0:
            return np.zeros_like(values, dtype="float32")

        # center on zero
        values -= col_min

        # apply a logarithm
        col_values = np.log(values + 1)

        # scale max to 1
        col_values *= 1. / np.log(col_range + 1)

        if self.clip_numerical_values:
            col_values = np.clip(col_values, 0., 1.)

        return col_values
    
    def fit_transform_numerical_df(self, df: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
        """
        Fits and transforms numerical columns in the given DataFrame.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            numerical_columns (list): List of numerical column names to transform.
            
        Returns:
            pd.DataFrame: A copy of the DataFrame with transformed numerical columns.
        """
        df_transformed = df.copy()

        for col in numerical_columns:
            self.__fit_numerical(col, df[col])
            df_transformed[col] = self.__transform_numerical(col, df[col])

        return df_transformed
    
    def __fit_categorical(self, column_name: str, n_categorical_levels, values: np.array):
        levels, level_counts = np.unique(values, return_counts=True)
        sorted_levels = list(sorted(zip(levels, level_counts), key=lambda x: x[1], reverse=True))
        self.encoded_levels[column_name] = [s[0] for s in sorted_levels[:n_categorical_levels]]


    def __transform_categorical(self, column_name:str, values: np.array, expected_categorical_format: str = "integer"):
        encoded_levels = self.encoded_levels[column_name]
        print(f"Encoding the {len(encoded_levels)} levels for {column_name}")

        result_values = np.ones(len(values), dtype="uint32")
        for level_i, level in enumerate(encoded_levels):
            level_mask = values == level

            # we use +1 here, as 0 = previously unseen, and 1 to (n + 1) are the encoded levels
            result_values[level_mask] = level_i + 1

        if expected_categorical_format == "integer":
            return result_values

        v = pd.get_dummies(result_values, prefix=column_name)
        return v
    
    
    def fit_transform_categoricals_df(self, df: pd.DataFrame, categorical_columns: list, n_categorical_levels: int, expected_categorical_format: str = "integer") -> pd.DataFrame:
        """
        Fits and transforms specified categorical columns in the DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            categorical_columns (list): List of categorical column names to transform.
            n_categorical_levels (int): Number of top categories to keep per column.
            expected_categorical_format (str): 'integer' or 'onehot' encoding output.
            
        Returns:
            pd.DataFrame: Transformed DataFrame with categorical columns replaced.
                        One-hot encoding will expand columns accordingly.
        """
        df_transformed = df.copy()

        for col in categorical_columns:
            # Fit top-N categories for the column
            self.__fit_categorical(col, n_categorical_levels, df_transformed[col].values)
            
            # Transform the column
            transformed = self.__transform_categorical(col, df_transformed[col].values, expected_categorical_format)
            
            # Drop the original column
            df_transformed.drop(columns=[col], inplace=True)
            
            # If integer encoded, add back as a single column
            if expected_categorical_format == "integer":
                df_transformed[col] = transformed
            else:
                # One-hot encoding returns a DataFrame, concat columns
                df_transformed = pd.concat([df_transformed, transformed], axis=1)

        return df_transformed
    
    def fit_transform_df_auto(self, df: pd.DataFrame, n_categorical_levels: int, expected_categorical_format: str = "integer") -> pd.DataFrame:
        """
        Automatically detects and preprocesses numerical and categorical columns.

        Parameters:
            df (pd.DataFrame): Input DataFrame (with features only, not target).
            n_categorical_levels (int): Number of top categories to retain per categorical column.
            expected_categorical_format (str): 'integer' or 'onehot'

        Returns:
            pd.DataFrame: Fully preprocessed DataFrame.
        """
        # Automatically detect column types
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        df_processed = df.copy()

        if numerical_columns:
            df_processed = self.fit_transform_numerical_df(df_processed, numerical_columns)

        if categorical_columns:
            df_processed = self.fit_transform_categoricals_df(
                df_processed,
                categorical_columns=categorical_columns,
                n_categorical_levels=n_categorical_levels,
                expected_categorical_format=expected_categorical_format
            )

        return df_processed

    

