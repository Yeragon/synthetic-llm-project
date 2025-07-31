import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ks_2samp, entropy
from src.evaluation import tstr_evaluate
from src.privacy import apply_differential_privacy
from src.metrics import compare_datasets


st.set_page_config(page_title="Real vs Synthetic Data Multi-field Analysis", layout="wide")
st.title("📊 Multi-field Comparison: Real Data vs Synthetic Data")

# Similarity Scoring Function 
def calculate_similarity_score(js_list, ks_list):
    valid_js = [x for x in js_list if x is not None and not pd.isna(x)]
    valid_ks = [x for x in ks_list if x is not None and not pd.isna(x)]

    if not valid_js and not valid_ks:
        return 0.0

    js_score = 1 - np.mean(valid_js) if valid_js else 0
    ks_score = np.mean(valid_ks) if valid_ks else 0

    final_score = 0.5 * js_score + 0.5 * ks_score
    return round(final_score * 100, 2)

# Data Paths
real_data_path = "data/adult.csv"
synthetic_data_dir = "data/synthetic"
synthetic_files = [f for f in os.listdir(synthetic_data_dir) if f.endswith(".csv")]

# Synthetic Data Selection
st.sidebar.title("📂 Data Selection")
selected_synthetic = st.sidebar.selectbox("Select synthetic data file", synthetic_files)

# Differential Privacy Control
use_privacy = st.sidebar.checkbox("🔐 Enable Differential Privacy")

# Load Data
real_df = pd.read_csv(real_data_path)
synthetic_df = pd.read_csv(os.path.join(synthetic_data_dir, selected_synthetic))

# Apply Differential Privacy
if use_privacy:
    synthetic_df = apply_differential_privacy(synthetic_df, epsilon=1)

# Align Columns
common_columns = [col for col in real_df.columns if col in synthetic_df.columns]
real_df = real_df[common_columns]
synthetic_df = synthetic_df[common_columns]

st.subheader("✅ Fields Aligned Successfully")
st.write(f"Number of aligned fields:{len(common_columns)} 个")
st.write("Field list:", common_columns)

# Use the metrics module to uniformly calculate the evaluation indicators of each field
metrics_df = compare_datasets(real_df, synthetic_df).set_index("column")

# Display Table
st.subheader("📉 Distribution Divergence Metrics (Lower is Better)")
numeric_cols = ['js_divergence', 'ks_p_value', 'mean_diff', 'std_diff']
styled_df = metrics_df.copy()
styled_numeric = styled_df[numeric_cols]
st.dataframe(styled_df.style.background_gradient(axis=0, subset=styled_numeric.columns, cmap='Reds'))

# Display Notes
if 'note' in metrics_df.columns and metrics_df['note'].notna().any():
    with st.expander("📌 Field Notes & Warnings"):
        st.dataframe(metrics_df[['type', 'note']][metrics_df['note'].notna()])


# Display Heatmap
st.subheader("🔥 Field Divergence Heatmap")

fig, ax = plt.subplots(figsize=(12, 6))

numeric_cols = ['js_divergence', 'ks_p_value', 'mean_diff', 'std_diff']
heatmap_df = metrics_df[numeric_cols].copy()

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_df.T, annot=True, cmap="Reds", cbar=True, fmt=".3f", ax=ax)
st.pyplot(fig)

# Auto Similarity Score
similarity_score = calculate_similarity_score(
    js_list=metrics_df["js_divergence"].tolist(),
    ks_list=metrics_df["ks_p_value"].tolist()
)
st.subheader("🧠 Automatic Similarity Scoring Report")
st.metric(label="Similarity Score Between Real and Synthetic Data", value=f"{similarity_score} / 100")
st.caption("Higher score indicates synthetic data is closer to real data (combined KS & JS scores)")
st.info("🔐 Differential Privacy is ENABLED" if use_privacy else "🔓 Differential Privacy is DISABLED")

# Download Report (CSV) 
st.subheader("📥 Download Report (CSV) ")
csv_data = metrics_df.to_csv().encode('utf-8')

st.download_button(
    label="📄 Download Divergence Report (CSV)",
    data=csv_data,
    file_name=f"diff_report_{selected_synthetic.replace('.csv', '')}.csv",
    mime="text/csv"
)

# Downstream Task TSTR Evaluation
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Downstream Task TSTR Evaluation")

# Select Target Column (Only Categorical Fields)
category_columns = [col for col in common_columns if real_df[col].dtype == 'object']
target_col = st.sidebar.selectbox("Select Target Column for Classification", category_columns)

# Run Evaluation Button
if st.sidebar.button("▶️ Run TSTR Evaluation"):
    try:
        tstr_result = tstr_evaluate(real_df, synthetic_df, target_col=target_col)
        st.session_state["tstr_result"] = tstr_result
        st.session_state["tstr_target"] = target_col
    except Exception as e:
        st.error(f"Evaluation failed:{e}")

# Display the assessment results (if already implemented)
if "tstr_result" in st.session_state:
    st.subheader(f"📈 Downstream Task Evaluation Result (TSTR) - Target Column:{st.session_state['tstr_target']}")
    st.json(st.session_state["tstr_result"])
