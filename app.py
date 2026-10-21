
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR + AUTOMATIC DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV stored beside app.py in the GitHub repository

def prepare_data(raw_df):
    """
    Clean and validate the Near-Earth Object dataset.
    Returns the cleaned DataFrame and an error message, if any.
    """
    df = raw_df.copy()

    # Normalize column names
    df.columns = df.columns.astype(str).str.strip()

    # Check for the target column
    if "pha" not in df.columns:
        return None, (
            "Required column 'pha' was not found. "
            "Please check the CSV column names."
        )

    # Standardize PHA labels
    df["pha"] = (
        df["pha"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[df["pha"].isin(["Y", "N"])].copy()

    if df.empty:
        return None, "No valid PHA labels (Y/N) were found in the dataset."

    # Convert relevant orbital features to numeric when present
    numeric_columns = [
        "a", "e", "q", "i", "om", "w", "ad", "n",
        "per", "per_y", "moid", "H"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.reset_index(drop=True, inplace=True)

    return df, None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "neos-asteroid dataset.csv"

try:
    raw_df = pd.read_csv(DATA_PATH, low_memory=False)

except FileNotFoundError:
    st.error(
        "Dataset not found. Please ensure that "
        "'neos-asteroid dataset.csv' exists in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Could not load the asteroid dataset: {e}")
    st.stop()

# Prepare and validate the dataset
# IMPORTANT: prepare_data() must be defined earlier in app.py
df, error = prepare_data(raw_df)

if error:
    st.error(error)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

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

    st.success(
        f"Dataset loaded automatically · {len(df):,} labelled objects"
    )

    st.caption("Source file: neos-asteroid dataset.csv")

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
        '<div class="small-muted">'
        'PHA = Potentially Hazardous Asteroid. '
        'This dashboard is an educational analysis tool, not an '
        'operational impact-warning system.'
        '</div>',
        unsafe_allow_html=True
    )
