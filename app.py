
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR + AUTOMATIC DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# Load the CSV stored beside app.py in the GitHub repository.
DATA_PATH = Path(__file__).parent / "neso.csv"

try:
    raw_df = pd.read_csv(DATA_PATH, low_memory=False)
except FileNotFoundError:
    st.error(
        "Dataset not found. Please ensure that neso.csv is uploaded "
        "to the same GitHub repository folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load neso.csv: {e}")
    st.stop()

# Prepare and validate the dataset
df, error = prepare_data(raw_df)

if error:
    st.error(error)
    st.stop()

# Sidebar controls
with st.sidebar:
    st.markdown(
        '<div class="eyebrow">ORBITAL / MISSION CONTROL</div>',
        unsafe_allow_html=True
    )
    st.markdown("## Data console")
    st.markdown(
        '<span class="tag">NEO · PHA CLASSIFICATION</span>',
        unsafe_allow_html=True
    )
    st.write("")

    st.success(f"Dataset loaded automatically · {len(df):,} labelled objects")
    st.caption("Source file: neso.csv")

    st.markdown("---")
    st.markdown("**Explore the catalogue**")
    st.caption("Filters update the charts and table immediately.")

    show_only = st.selectbox(
        "PHA class",
        [
            "All objects",
            "Potentially hazardous (Y)",
            "Not classified as PHA (N)"
        ]
    )

    st.markdown("---")
    st.markdown("**Reading the risk labels**")
    st.markdown(
        '<div class="small-muted">PHA = Potentially Hazardous Asteroid. '
        'This dashboard is an educational analysis tool, not an '
        'operational impact-warning system.</div>',
        unsafe_allow_html=True
    )
