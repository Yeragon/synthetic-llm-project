import pandas as pd
import numpy as np
from scipy.stats import entropy, ks_2samp
import warnings

def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def _numeric_stats(real_col: pd.Series, synth_col: pd.Series):
    # 共用 bin，避免边界影响
    vals = np.concatenate([real_col.values, synth_col.values])
    bins = np.histogram_bin_edges(vals, bins="auto")
    ph, _ = np.histogram(real_col.values, bins=bins)
    qh, _ = np.histogram(synth_col.values, bins=bins)
    js = js_divergence(ph, qh)
    ks = ks_2samp(real_col.values, synth_col.values, alternative="two-sided", mode="auto")
    mean_diff = float(real_col.mean() - synth_col.mean())        # 有符号差
    std_diff  = float(real_col.std(ddof=0) - synth_col.std(ddof=0))
    return js, float(ks.statistic), float(ks.pvalue), mean_diff, std_diff, None

def _categorical_stats(real_col: pd.Series, synth_col: pd.Series):
    va = real_col.value_counts(dropna=False, normalize=True)
    vb = synth_col.value_counts(dropna=False, normalize=True)
    cats = sorted(set(va.index) | set(vb.index))
    pa = np.array([va.get(c, 0.0) for c in cats])
    pb = np.array([vb.get(c, 0.0) for c in cats])
    js = js_divergence(pa, pb)
    note = None
    if set(va.index) != set(vb.index):
        add = len(set(vb.index) - set(va.index))
        miss = len(set(va.index) - set(vb.index))
        note = f"category_mismatch:+{add}/-{miss}"
    return js, np.nan, np.nan, np.nan, np.nan, note

def compute_column_stats(real_col, synth_col, col_name):
    # 统一丢 NA
    a = real_col.dropna()
    b = synth_col.dropna()

    out = {
        "column": col_name,
        "type": str(real_col.dtype),
        "js_divergence": np.nan,
        "ks_stat": np.nan,
        "ks_p_value": np.nan,
        "mean_diff": np.nan,
        "std_diff": np.nan,
        "note": ""
    }

    if len(a) == 0 or len(b) == 0:
        out["note"] = "empty_after_dropna"
        return out

    # 数值/类别双判定 + 轻度尝试类型对齐
    is_num_a = pd.api.types.is_numeric_dtype(a)
    is_num_b = pd.api.types.is_numeric_dtype(b)
    if is_num_a and not is_num_b:
        # 尝试把 b 转成数值
        b_try = pd.to_numeric(b, errors="coerce").dropna()
        if len(b_try) > 0:
            b = b_try
            is_num_b = True
    if not is_num_a and is_num_b:
        a_try = pd.to_numeric(a, errors="coerce").dropna()
        if len(a_try) > 0:
            a = a_try
            is_num_a = True

    if is_num_a and is_num_b:
        if a.min() == a.max() and b.min() == b.max():
            out["note"] = "constant_columns"
            return out
        js, ks_stat, ks_p, md, sd, note = _numeric_stats(a.astype(float), b.astype(float))
        out.update({
            "type": "number",
            "js_divergence": js,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p,
            "mean_diff": md,
            "std_diff": sd,
            "note": note or ""
        })
    else:
        js, ks_stat, ks_p, md, sd, note = _categorical_stats(a.astype(str), b.astype(str))
        out.update({
            "type": "object",
            "js_divergence": js,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p,
            "mean_diff": md,
            "std_diff": sd,
            "note": note or ""
        })
    return out

def compare_datasets(real_df, synth_df, verbose=False):
    rows = []
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    if verbose:
        print(f"Common columns: {common_cols}")
    for col in common_cols:
        try:
            rows.append(compute_column_stats(real_df[col], synth_df[col], col))
        except Exception as e:
            warnings.warn(f"Error processing column {col}: {e}")
            rows.append({
                "column": col, "type": str(real_df[col].dtype),
                "js_divergence": np.nan, "ks_stat": np.nan, "ks_p_value": np.nan,
                "mean_diff": np.nan, "std_diff": np.nan, "note": f"error:{e}"
            })
    return pd.DataFrame(rows)
