import pandas as pd
import numpy as np
from scipy.stats import entropy, ks_2samp
import warnings

def js_divergence(p, q, eps=1e-10):
    """
    Compute Jensen-Shannon Divergence between two probability distributions.
    """
    p = np.array(p) + eps
    q = np.array(q) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def compute_column_stats(real_col, synth_col, col_name, bins=20):
    """
    Compute similarity metrics between two columns: JS divergence, KS test, mean & std diff.
    """
    stats = {
        'column': col_name,
        'type': str(real_col.dtype),
        'js_divergence': np.nan,
        'ks_p_value': np.nan,
        'mean_diff': np.nan,
        'std_diff': np.nan,
        'note': ''
    }

    # Drop NA
    real_col = real_col.dropna()
    synth_col = synth_col.dropna()

    if len(real_col) == 0 or len(synth_col) == 0:
        stats['note'] = 'Missing or empty data'
        return stats

    if pd.api.types.is_numeric_dtype(real_col):
        min_val = min(real_col.min(), synth_col.min())
        max_val = max(real_col.max(), synth_col.max())

        if min_val == max_val:
            stats['note'] = 'Constant value'
            return stats

        real_hist, bin_edges = np.histogram(real_col, bins=bins, range=(min_val, max_val), density=True)
        synth_hist, _ = np.histogram(synth_col, bins=bin_edges, density=True)

        stats['js_divergence'] = js_divergence(real_hist, synth_hist)
        stats['ks_p_value'] = ks_2samp(real_col, synth_col).pvalue
        stats['mean_diff'] = abs(real_col.mean() - synth_col.mean())
        stats['std_diff'] = abs(real_col.std() - synth_col.std())

    else:
        # Categorical
        real_freq = real_col.value_counts(normalize=True)
        synth_freq = synth_col.value_counts(normalize=True)
        all_cats = sorted(set(real_freq.index).union(set(synth_freq.index)))

        real_vec = [real_freq.get(cat, 0) for cat in all_cats]
        synth_vec = [synth_freq.get(cat, 0) for cat in all_cats]

        stats['js_divergence'] = js_divergence(real_vec, synth_vec)

    return stats

def compare_datasets(real_df, synth_df, verbose=False):
    """
    Compare two datasets column-by-column and compute similarity metrics.
    """
    results = []
    common_cols = [col for col in real_df.columns if col in synth_df.columns]

    if verbose:
        print(f"Common columns found: {common_cols}")

    for col in common_cols:
        try:
            result = compute_column_stats(real_df[col], synth_df[col], col)
            results.append(result)
        except Exception as e:
            warnings.warn(f"Error processing column {col}: {e}")
            results.append({'column': col, 'type': str(real_df[col].dtype), 'error': str(e)})

    return pd.DataFrame(results)
