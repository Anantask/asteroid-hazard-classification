
# =============================================================================
# ORBITAL INTELLIGENCE
# Near-Earth Object | Potentially Hazardous Asteroid Classification
# Streamlit dashboard | Dark astronomy theme
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ORBITAL | NEO Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# PROFESSIONAL DARK ASTRONOMY THEME
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d18;
    --panel: #101a2b;
    --panel2: #131f33;
    --line: #253650;
    --text: #e8effa;
    --muted: #93a6c1;
    --cyan: #62d9f5;
    --blue: #709bff;
    --amber: #f2bd72;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 82% 0%, rgba(49, 87, 143, .15), transparent 38%),
        radial-gradient(ellipse at 0% 35%, rgba(36, 111, 148, .08), transparent 34%),
        #080d18;
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8,13,24,.94);
}

[data-testid="stSidebar"] {
    background: #0b1321;
    border-right: 1px solid #253650;
}

[data-testid="stSidebar"] * {
    color: #dce7f7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1600px;
}

h1, h2, h3 {
    color: #edf4ff !important;
    letter-spacing: -0.6px;
}

p, label, .stMarkdown {
    color: #d3deed;
}

.orb-eyebrow {
    color: #62d9f5;
    font-family: 'DM Mono', monospace;
    letter-spacing: 3px;
    font-size: 11px;
    text-transform: uppercase;
}

.orb-hero {
    background: linear-gradient(115deg, #101b2e 0%, #111f35 54%, #102738 100%);
    border: 1px solid #2a405c;
    border-radius: 20px;
    padding: 30px 32px;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
}

.orb-hero:after {
    content: '';
    position: absolute;
    width: 210px;
    height: 210px;
    right: 45px;
    top: -100px;
    border: 1px solid rgba(98,217,245,.24);
    border-radius: 50%;
    box-shadow: 0 0 0 24px rgba(98,217,245,.035),
                0 0 0 50px rgba(98,217,245,.025);
    pointer-events: none;
}

.orb-title {
    font-size: clamp(30px, 4vw, 47px);
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -1.7px;
    color: #f1f6ff;
    margin: 8px 0 12px;
}

.orb-subtitle {
    color: #a7b9d1;
    font-size: 14px;
    line-height: 1.8;
    max-width: 780px;
}

.orb-chip {
    display: inline-block;
    border: 1px solid #34516b;
    border-radius: 5px;
    padding: 5px 9px;
    color: #80e4f8;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    margin: 0 6px 6px 0;
}

.orb-card {
    background: linear-gradient(145deg, #111d30, #0e1727);
    border: 1px solid #263a55;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 122px;
}

.orb-card-label {
    color: #96abc6;
    font-size: 11px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.orb-card-value {
    color: #f0f5ff;
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
}

.orb-card-note {
    color: #90a6c2;
    font-size: 11px;
    margin-top: 7px;
}

.orb-section {
    border-bottom: 1px solid #253650;
    padding-bottom: 10px;
    margin: 30px 0 18px;
}

.orb-section small {
    color: #62d9f5;
    font-family: 'DM Mono', monospace;
    letter-spacing: 2px;
}

.orb-panel {
    background: #0f1929;
    border: 1px solid #253650;
    border-radius: 14px;
    padding: 18px;
}

.orb-footer {
    border-top: 1px solid #253650;
    margin-top: 35px;
    padding-top: 18px;
    color: #8297b3;
    font-size: 11px;
    line-height: 1.8;
}

div[data-testid="stMetric"] {
    background: #111d30;
    border: 1px solid #263a55;
    padding: 15px 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #9db0ca;
}

div[data-testid="stMetricValue"] {
    color: #edf5ff;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #263a55;
}

.stTabs [data-baseweb="tab"] {
    color: #a7b9d1;
    padding: 12px 15px;
}

.stTabs [aria-selected="true"] {
    color: #62d9f5 !important;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid #34516b;
    background: #14263b;
    color: #dff7ff;
}

.stButton > button:hover {
    border-color: #62d9f5;
    color: #62d9f5;
}

.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div {
    background: #111d30;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263a55;
    border-radius: 10px;
}

hr {
    border-color: #253650;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "neos-asteroid dataset.csv"

PLOT_TEMPLATE = "plotly_dark"

PLOT_LAYOUT = dict(
    template=PLOT_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Manrope, sans-serif", color="#c8d6e9", size=11),
    margin=dict(l=18, r=18, t=55, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# Deliberately excludes MOID and H to reduce target leakage.
BASE_FEATURES = [
    "a", "e", "q", "i", "ad", "n", "per_y",
    "om_sin", "om_cos", "w_sin", "w_cos"
]

# =============================================================================
# DATA PREPARATION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_dataset(path_string, modified_time):
    return pd.read_csv(path_string, low_memory=False)


def prepare_data(raw_df):
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # Support common variants of the PHA target-column name.
    target_lookup = {c.lower(): c for c in df.columns}
    target_col = target_lookup.get("pha")

    if target_col is None:
        raise ValueError(
            "The CSV must contain a 'pha' column with Y/N labels. "
            f"Available columns: {', '.join(df.columns[:25])}"
        )

    if target_col != "pha":
        df.rename(columns={target_col: "pha"}, inplace=True)

    df["pha"] = (
        df["pha"].astype("string").str.strip().str.upper()
    )

    df = df[df["pha"].isin(["Y", "N"])].copy()

    if df.empty:
        raise ValueError("No labelled PHA rows with Y/N values were found.")

    # Numeric orbital parameters.
    for col in [
        "a", "e", "q", "i", "om", "w", "ad", "n",
        "per", "per_y", "moid", "H", "diameter",
        "albedo", "moid_ld", "sigma_e", "sigma_a"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Circular encoding prevents angular wrap-around problems.
    if "om" in df.columns:
        angle = np.deg2rad(df["om"])
        df["om_sin"] = np.sin(angle)
        df["om_cos"] = np.cos(angle)

    if "w" in df.columns:
        angle = np.deg2rad(df["w"])
        df["w_sin"] = np.sin(angle)
        df["w_cos"] = np.cos(angle)

    # Derive orbital period in years when possible.
    if "per_y" not in df.columns and "per" in df.columns:
        df["per_y"] = df["per"] / 365.25

    return df.reset_index(drop=True)


# =============================================================================
# MACHINE-LEARNING PIPELINE
# =============================================================================

def get_available_features(df):
    return [col for col in BASE_FEATURES if col in df.columns]


@st.cache_resource(show_spinner="Training the asteroid classification model...")
def train_model(data, features):
    """
    Train and evaluate a HistGradientBoosting classifier on labelled data.
    The held-out test set is not used to fit the model.
    """
    X = data[features].copy()
    y = (data["pha"] == "Y").astype(int)

    # Exclude rows with no usable input features.
    usable = X.notna().any(axis=1)
    X = X.loc[usable]
    y = y.loc[usable]

    if len(X) < 50 or y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "Not enough usable examples from both PHA classes to train "
            "the model. Check the orbital columns and labels."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Illustrative operating threshold, adjustable in the UI.
    default_threshold = 0.1612
    predictions = (probabilities >= default_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "baseline_rate": float(y_test.mean()),
        "test_size": len(y_test),
        "train_size": len(y_train),
    }

    test_results = pd.DataFrame({
        "Actual PHA": y_test.to_numpy(),
        "Predicted probability": probabilities,
        "Predicted PHA": predictions,
    })

    # Majority-class baseline for context.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    metrics["baseline_accuracy"] = accuracy_score(
        y_test, baseline_predictions
    )

    return model, metrics, test_results, X_test, y_test


# =============================================================================
# LOAD DATA
# =============================================================================

if not DATA_PATH.exists():
    st.error(
        "Dataset file not found. Upload **neos-asteroid dataset.csv** "
        "to the same GitHub repository folder as app.py."
    )
    st.stop()

try:
    raw_df = load_dataset(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime
    )
    df = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Could not prepare the asteroid dataset: {exc}")
    st.stop()

FEATURES = get_available_features(df)

if len(FEATURES) < 3:
    st.error(
        "The dataset does not contain enough supported orbital features "
        "for the model. Check that columns such as a, e, q and i exist."
    )
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('<div class="orb-eyebrow">ORBITAL / MISSION CONTROL</div>',
                unsafe_allow_html=True)
    st.markdown("## Data console")
    st.caption("NEAR-EARTH OBJECT INTELLIGENCE")
    st.markdown("---")

    st.markdown("**Catalogue filters**")

    pha_filter = st.selectbox(
        "PHA classification",
        ["All objects", "Potentially hazardous (Y)",
         "Not classified as PHA (N)"]
    )

    search_text = st.text_input(
        "Search object name / designation",
        placeholder="Enter an asteroid name..."
    )

    numeric_filter = st.selectbox(
        "Orbital parameter",
        [c for c in ["a", "e", "q", "i", "moid", "H"] if c in df.columns],
        index=0
    )

    if df[numeric_filter].notna().any():
        min_val = float(df[numeric_filter].min())
        max_val = float(df[numeric_filter].max())

        if min_val < max_val:
            range_value = st.slider(
                f"Range for {numeric_filter}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
        else:
            range_value = (min_val, max_val)
    else:
        range_value = None

    st.markdown("---")
    st.markdown("**Model controls**")
    threshold = st.slider(
        "PHA probability threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.1612,
        step=0.01,
        help="Lower thresholds flag more objects. This is a model setting, "
             "not an official hazard criterion."
    )

    st.markdown("---")
    st.markdown("**Data source**")
    st.caption("NASA/JPL NEO catalogue dataset supplied with this project.")
    st.caption(f"{len(df):,} labelled rows loaded.")
    st.caption("Educational analysis only; not an operational warning system.")

# =============================================================================
# FILTERED DATA
# =============================================================================

filtered = df.copy()

if pha_filter == "Potentially hazardous (Y)":
    filtered = filtered[filtered["pha"] == "Y"]
elif pha_filter == "Not classified as PHA (N)":
    filtered = filtered[filtered["pha"] == "N"]

if search_text.strip():
    text_columns = [
        c for c in ["name", "full_name", "designation", "pdes", "spkid"]
        if c in filtered.columns
    ]
    if text_columns:
        mask = pd.Series(False, index=filtered.index)
        for col in text_columns:
            mask |= filtered[col].astype(str).str.contains(
                search_text.strip(), case=False, na=False, regex=False
            )
        filtered = filtered[mask]

if range_value is not None:
    filtered = filtered[
        filtered[numeric_filter].between(
            range_value[0], range_value[1], inclusive="both"
        )
    ]

# =============================================================================
# HERO
# =============================================================================

st.markdown("""
<div class="orb-hero">
    <div class="orb-eyebrow">DEEP SPACE ANALYTICS &nbsp; / &nbsp; NEO-01</div>
    <div class="orb-title">ORBITAL <span style="color:#62d9f5;">INTELLIGENCE</span></div>
    <div class="orb-subtitle">
        An interactive observatory for exploring near-Earth objects,
        examining orbital characteristics, and evaluating a machine-learning
        model for Potentially Hazardous Asteroid (PHA) classification.
    </div>
    <br>
    <span class="orb-chip">ASTEROID CATALOGUE</span>
    <span class="orb-chip">ORBITAL DYNAMICS</span>
    <span class="orb-chip">MACHINE LEARNING</span>
    <span class="orb-chip">INTERACTIVE ANALYTICS</span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="orb-section"><small>01 / CATALOGUE OVERVIEW</small></div>',
    unsafe_allow_html=True
)

total_labelled = len(df)
total_pha = int((df["pha"] == "Y").sum())
total_nonpha = int((df["pha"] == "N").sum())
pha_rate = total_pha / total_labelled * 100 if total_labelled else 0

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="orb-card">
        <div class="orb-card-label">Labelled objects</div>
        <div class="orb-card-value">{total_labelled:,}</div>
        <div class="orb-card-note">Usable PHA-labelled records</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="orb-card">
        <div class="orb-card-label">PHA labelled (Y)</div>
        <div class="orb-card-value" style="color:#f2bd72;">{total_pha:,}</div>
        <div class="orb-card-note">Positive catalogue labels</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="orb-card">
        <div class="orb-card-label">PHA label share</div>
        <div class="orb-card-value">{pha_rate:.2f}%</div>
        <div class="orb-card-note">Share of labelled objects</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="orb-card">
        <div class="orb-card-label">Visible after filters</div>
        <div class="orb-card-value">{len(filtered):,}</div>
        <div class="orb-card-note">Current catalogue selection</div>
    </div>""", unsafe_allow_html=True)

st.caption(
    "Catalogue label counts describe this dataset; they are not predictions "
    "of imminent impacts."
)

# =============================================================================
# MODEL TRAINING
# =============================================================================

try:
    model, model_metrics, test_results, X_test, y_test = train_model(
        df, tuple(FEATURES)
    )
    model_error = None
except Exception as exc:
    model_error = str(exc)
    model = None
    model_metrics = {}
    test_results = pd.DataFrame()
    X_test = None
    y_test = None

# =============================================================================
# MAIN TABS
# =============================================================================

tab_overview, tab_orbits, tab_ml, tab_catalogue, tab_method = st.tabs([
    "◈  Observatory",
    "◎  Orbital Analysis",
    "⌘  ML Intelligence",
    "▤  Catalogue Explorer",
    "ⓘ  Methodology",
])

# =============================================================================
# TAB 1: OBSERVATORY
# =============================================================================

with tab_overview:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("PHA label distribution")
        counts = df["pha"].value_counts().reindex(["Y", "N"], fill_value=0)

        fig = go.Figure(data=[go.Pie(
            labels=["Potentially hazardous (Y)", "Not classified as PHA (N)"],
            values=[int(counts["Y"]), int(counts["N"])],
            hole=0.72,
            sort=False,
            marker=dict(
                colors=["#f2bd72", "#62d9f5"],
                line=dict(color="#0b1321", width=4)
            ),
            textinfo="percent",
            textfont=dict(size=13),
            hovertemplate="%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        )])
        fig.update_layout(**PLOT_LAYOUT, height=340, showlegend=True)
        fig.add_annotation(
            text=f"<b>{total_labelled:,}</b><br><sup>labelled</sup>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=21, color="#edf5ff")
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Orbital parameter coverage")

        coverage_cols = [
            c for c in ["a", "e", "q", "i", "om", "w", "ad", "n", "per_y"]
            if c in df.columns
        ]

        coverage = pd.DataFrame({
            "Parameter": coverage_cols,
            "Available (%)": [
                df[c].notna().mean() * 100 for c in coverage_cols
            ]
        }).sort_values("Available (%)", ascending=True)

        fig = px.bar(
            coverage,
            x="Available (%)",
            y="Parameter",
            orientation="h",
            range_x=[0, 100],
            color="Available (%)",
            color_continuous_scale=["#233b58", "#62d9f5"],
        )
        fig.update_layout(
            **PLOT_LAYOUT, height=340, coloraxis_showscale=False,
            xaxis_title="Records with a value (%)", yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Orbital landscape")

    if all(c in df.columns for c in ["a", "e"]):
        scatter_df = filtered.dropna(subset=["a", "e"]).copy()
        scatter_df["PHA label"] = scatter_df["pha"].map({
            "Y": "PHA label: Y",
            "N": "PHA label: N"
        })

        hover_cols = [
            c for c in ["name", "full_name", "pdes", "q", "i", "moid", "H"]
            if c in scatter_df.columns
        ]

        fig = px.scatter(
            scatter_df,
            x="a",
            y="e",
            color="PHA label",
            color_discrete_map={
                "PHA label: Y": "#f2bd72",
                "PHA label: N": "#62d9f5"
            },
            hover_data=hover_cols,
            opacity=0.7,
            render_mode="webgl",
            labels={
                "a": "Semi-major axis (AU)",
                "e": "Orbital eccentricity"
            },
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(**PLOT_LAYOUT, height=480)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The dataset does not contain both a and e orbital columns.")

# =============================================================================
# TAB 2: ORBITAL ANALYSIS
# =============================================================================

with tab_orbits:
    st.subheader("Explore orbital characteristics")
    st.caption("Select parameters to compare their distributions across catalogue labels.")

    available_numeric = [
        c for c in [
            "a", "e", "q", "i", "om", "w", "ad", "n",
            "per_y", "moid", "H", "diameter", "albedo"
        ]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]

    if len(available_numeric) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox(
                "Horizontal axis", available_numeric,
                index=available_numeric.index("a") if "a" in available_numeric else 0,
                key="orbit_x"
            )
        with c2:
            y_axis = st.selectbox(
                "Vertical axis", available_numeric,
                index=available_numeric.index("e") if "e" in available_numeric else 1,
                key="orbit_y"
            )

        orbit_df = filtered.dropna(subset=[x_axis, y_axis]).copy()
        orbit_df["PHA label"] = orbit_df["pha"].map({
            "Y": "PHA label: Y",
            "N": "PHA label: N"
        })

        fig = px.scatter(
            orbit_df,
            x=x_axis, y=y_axis,
            color="PHA label",
            color_discrete_map={
                "PHA label: Y": "#f2bd72",
                "PHA label: N": "#62d9f5"
            },
            opacity=0.68,
            render_mode="webgl",
            hover_data=[
                c for c in ["name", "pdes", "q", "i", "moid", "H"]
                if c in orbit_df.columns and c not in [x_axis, y_axis]
            ],
        )
        fig.update_traces(marker=dict(size=6))
        fig.update_layout(**PLOT_LAYOUT, height=480)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Distribution of {x_axis}")
        hist_df = filtered.dropna(subset=[x_axis]).copy()
        hist_df["PHA label"] = hist_df["pha"].map({
            "Y": "PHA label: Y",
            "N": "PHA label: N"
        })
        fig = px.histogram(
            hist_df, x=x_axis, color="PHA label", barmode="overlay",
            opacity=0.72, marginal="box",
            color_discrete_map={
                "PHA label: Y": "#f2bd72",
                "PHA label: N": "#62d9f5"
            },
        )
        fig.update_layout(**PLOT_LAYOUT, height=420)
        st.plotly_chart(fig, use_container_width=True)

        if len(available_numeric) >= 3:
            st.subheader("Orbital feature correlation")
            corr_cols = [
                c for c in available_numeric
                if df[c].nunique(dropna=True) > 1
            ][:12]
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr(method="spearman")
                fig = px.imshow(
                    corr, text_auto=".2f",
                    color_continuous_scale=[
                        "#17243a", "#385c7c", "#62d9f5", "#f2bd72"
                    ],
                    zmin=-1, zmax=1, aspect="auto"
                )
                fig.update_layout(**PLOT_LAYOUT, height=520)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric orbital parameters are available.")

# =============================================================================
# TAB 3: ML INTELLIGENCE
# =============================================================================

with tab_ml:
    st.subheader("Machine-learning intelligence")
    st.caption(
        "HistGradientBoosting classifier | 80/20 stratified hold-out split | "
        "orbital features only"
    )

    if model_error:
        st.error(f"Model could not be trained: {model_error}")
    else:
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("ROC-AUC", f"{model_metrics['roc_auc']:.3f}")
        with m2:
            st.metric("PR-AUC / Average precision",
                      f"{model_metrics['pr_auc']:.3f}")
        with m3:
            st.metric("Recall", f"{model_metrics['recall']:.3f}")
        with m4:
            st.metric("F1 score", f"{model_metrics['f1']:.3f}")

        st.info(
            f"The model was evaluated on {model_metrics['test_size']:,} held-out "
            f"records after training on {model_metrics['train_size']:,} records. "
            f"The displayed default operating threshold is 0.1612; the sidebar "
            f"lets you explore how changing it affects the test-set metrics. "
            f"These are retrospective dataset results, not verified future impact predictions."
        )

        current_predictions = (
            test_results["Predicted probability"].to_numpy() >= threshold
        ).astype(int)
        actual = test_results["Actual PHA"].to_numpy()

        p1, p2 = st.columns(2, gap="large")

        with p1:
            st.markdown("#### Confusion matrix")
            cm = confusion_matrix(actual, current_predictions, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual N", "Actual Y"],
                columns=["Predicted N", "Predicted Y"]
            )
            fig = px.imshow(
                cm_df, text_auto=True,
                color_continuous_scale=["#101a2b", "#246b83", "#62d9f5"],
                aspect="auto"
            )
            fig.update_layout(**PLOT_LAYOUT, height=370)
            st.plotly_chart(fig, use_container_width=True)

        with p2:
            st.markdown("#### Metrics at selected threshold")
            threshold_metrics = pd.DataFrame({
                "Metric": ["Precision", "Recall", "F1 score", "Accuracy"],
                "Value": [
                    precision_score(actual, current_predictions, zero_division=0),
                    recall_score(actual, current_predictions, zero_division=0),
                    f1_score(actual, current_predictions, zero_division=0),
                    accuracy_score(actual, current_predictions)
                ]
            })
            fig = px.bar(
                threshold_metrics, x="Metric", y="Value",
                text=threshold_metrics["Value"].map(lambda v: f"{v:.3f}"),
                range_y=[0, 1],
                color="Metric",
                color_discrete_sequence=[
                    "#62d9f5", "#709bff", "#f2bd72", "#7ed6b1"
                ]
            )
            fig.update_layout(
                **PLOT_LAYOUT, height=370, showlegend=False,
                yaxis_title="Score", xaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Model performance compared with a simple baseline")
        baseline_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC", "PR-AUC"],
            "Model": [
                accuracy_score(actual, current_predictions),
                model_metrics["roc_auc"],
                model_metrics["pr_auc"]
            ],
            "Majority-class baseline": [
                model_metrics["baseline_accuracy"],
                0.5,
                model_metrics["baseline_rate"]
            ]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=baseline_df["Metric"], y=baseline_df["Model"],
            name="Classifier", marker_color="#62d9f5"
        ))
        fig.add_trace(go.Bar(
            x=baseline_df["Metric"],
            y=baseline_df["Majority-class baseline"],
            name="Simple baseline", marker_color="#536985"
        ))
        fig.update_layout(**PLOT_LAYOUT, barmode="group", height=350,
                          yaxis_title="Score", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Explore model scores on the held-out test set")
        st.caption(
            "Changing the threshold updates predicted labels, confusion matrix, "
            "precision, recall, F1 and accuracy. The underlying model remains fixed."
        )

        score_df = test_results.copy()
        score_df["Actual label"] = score_df["Actual PHA"].map({
            0: "Actual N", 1: "Actual Y"
        })
        score_df["Predicted label at threshold"] = np.where(
            score_df["Predicted probability"] >= threshold,
            "Predicted Y", "Predicted N"
        )

        fig = px.histogram(
            score_df,
            x="Predicted probability",
            color="Actual label",
            nbins=35,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={
                "Actual N": "#62d9f5",
                "Actual Y": "#f2bd72"
            },
        )
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="#ffffff",
            annotation_text=f"Threshold {threshold:.2f}"
        )
        fig.update_layout(**PLOT_LAYOUT, height=390)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Inspect individual test-set records")
        score_df["Actual PHA"] = score_df["Actual PHA"].map({0: "N", 1: "Y"})
        score_df["Predicted PHA"] = np.where(
            score_df["Predicted probability"] >= threshold, "Y", "N"
        )
        st.dataframe(
            score_df.head(100).style.format({
                "Predicted probability": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The probability is a model score learned from this dataset. "
            "It is not an official NASA impact probability or a validated "
            "measure of real-world collision risk."
        )

# =============================================================================
# TAB 4: CATALOGUE EXPLORER
# =============================================================================

with tab_catalogue:
    st.subheader("Asteroid catalogue")
    st.caption(
        "Use the sidebar filters to narrow the records. Sort and explore "
        "the available orbital properties below."
    )

    st.markdown(f"**{len(filtered):,} records** match the current filters.")

    preferred_columns = [
        "name", "full_name", "pdes", "pha", "neo",
        "a", "e", "q", "i", "moid", "H", "diameter",
        "albedo", "per_y", "n"
    ]
    display_columns = [c for c in preferred_columns if c in filtered.columns]

    if not display_columns:
        display_columns = list(filtered.columns[:15])

    st.dataframe(
        filtered[display_columns].head(2000),
        use_container_width=True,
        hide_index=True,
        height=500
    )

    st.caption(
        "For performance, the table displays at most 2,000 matching rows. "
        "The summary metrics and charts use the complete filtered selection."
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="↓  Download filtered catalogue (CSV)",
        data=csv_bytes,
        file_name="orbital_filtered_asteroid_catalogue.csv",
        mime="text/csv"
    )

    st.markdown("#### Export summary")
    export_summary = pd.DataFrame({
        "Metric": [
            "Labelled objects in source dataset",
            "PHA-labelled objects (Y)",
            "Not classified as PHA (N)",
            "Rows matching current filters",
            "PHA share in source dataset (%)"
        ],
        "Value": [
            total_labelled,
            total_pha,
            total_nonpha,
            len(filtered),
            round(pha_rate, 3)
        ]
    })
    st.dataframe(export_summary, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 5: METHODOLOGY
# =============================================================================

with tab_method:
    st.subheader("Methodology & scientific context")

    st.markdown("""
    <div class="orb-panel">
      <div class="orb-eyebrow">DATA PIPELINE</div>
      <h3>From orbital records to model evaluation</h3>
      <p style="color:#a7b9d1;line-height:1.9">
        The dashboard reads the repository's labelled asteroid catalogue,
        retains records whose PHA target is Y or N, converts selected orbital
        fields to numeric values, and encodes angular variables as sine/cosine
        pairs to account for the circular nature of angles.
      </p>
      <p style="color:#a7b9d1;line-height:1.9">
        A HistGradientBoosting classifier is trained on a stratified 80% split
        and evaluated on a held-out 20% split. Missing feature values are
        imputed using training-set medians inside a pipeline. MOID and absolute
        magnitude H are excluded from the selected model features to reduce
        possible target leakage and encourage analysis of orbital inputs.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Selected model inputs")
    st.write(", ".join(FEATURES))

    st.markdown("#### What the metrics mean")
    definitions = pd.DataFrame({
        "Metric": [
            "Accuracy", "Precision", "Recall", "F1 score",
            "ROC-AUC", "PR-AUC / Average precision"
        ],
        "Interpretation": [
            "Fraction of test predictions that are correct.",
            "Among predicted PHA cases, the fraction labelled PHA.",
            "Among actual PHA cases in the test set, the fraction detected.",
            "Harmonic mean of precision and recall.",
            "Ability to rank positive examples above negative examples across thresholds.",
            "Precision-recall performance; especially informative with imbalanced classes."
        ]
    })
    st.dataframe(definitions, use_container_width=True, hide_index=True)

    st.warning(
        "Scientific limitation: a catalogue's PHA label is not equivalent to "
        "an imminent impact prediction. This educational classifier is not a "
        "substitute for official orbit determination, uncertainty analysis, "
        "or NASA/JPL impact-monitoring systems."
    )

    st.markdown("#### Dataset information")
    info = pd.DataFrame({
        "Property": [
            "Source file", "Raw rows", "Labelled rows used",
            "Columns", "PHA Y count", "PHA N count",
            "Selected numeric features"
        ],
        "Value": [
            DATA_PATH.name, f"{len(raw_df):,}", f"{len(df):,}",
            str(len(raw_df.columns)), f"{total_pha:,}", f"{total_nonpha:,}",
            str(len(FEATURES))
        ]
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div class="orb-footer">
    <b style="color:#b8cce5;">ORBITAL INTELLIGENCE</b>
    &nbsp; / &nbsp; Near-Earth Object Analytics<br>
    Educational machine-learning and data-visualization project.
    Catalogue labels and model scores are for analysis only and should not be
    interpreted as official impact-risk assessments.<br>
    <span style="font-family:'DM Mono',monospace;color:#62d9f5;">
    OBSERVE &nbsp;·&nbsp; ANALYSE &nbsp;·&nbsp; UNDERSTAND
    </span>
</div>
""", unsafe_allow_html=True)
