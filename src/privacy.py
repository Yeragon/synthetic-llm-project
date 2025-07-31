import numpy as np
import pandas as pd

def apply_differential_privacy(df, epsilon=1, sensitivity=1.0):
    """
    Apply Laplace noise to numeric columns for differential privacy.
    """
    noisy_df = df.copy()
    numeric_cols = noisy_df.select_dtypes(include=[np.number]).columns
    scale = sensitivity / epsilon

    for col in numeric_cols:
        noise = np.random.laplace(0, scale, noisy_df[col].shape)
        noisy_df[col] = noisy_df[col] + noise

    return noisy_df