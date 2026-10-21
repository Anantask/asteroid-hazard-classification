import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE + VISUAL SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ORBITAL | Near-Earth Asteroid Intelligence",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#080D1A"
PANEL = "#10192B"
PANEL_2 = "#121F35"
TEXT = "#F1F5FF"
MUTED = "#98A8C7"
CYAN = "#6CE5E8"
AMBER = "#FFBE69"
CORAL = "#FF7F86"
VIOLET = "#B8A5FF"
GREEN = "#83E6B2"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {{ color-scheme: dark; }}
      html, body, [class*="css"] {{ font-family: 'Manrope', sans-serif; }}
      .stApp {{ background: radial-gradient(ellipse at 80% -10%, #182948 0%, {BG} 47%); color: {TEXT}; }}
      [data-testid="stHeader"] {{ background: rgba(8,13,26,.72); }}
      [data-testid="stSidebar"] {{ background: #0B1323; border-right: 1px solid #1C2A43; }}
      [data-testid="stSidebar"] * {{ color: {TEXT}; }}
      .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
      h1,h2,h3 {{ letter-spacing: -.035em; color: {TEXT}; }}
      p, label, .stMarkdown {{ color: {TEXT}; }}
      .eyebrow {{ color: {CYAN}; font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; }}
      .hero {{ position: relative; overflow: hidden; padding: 1.65rem 1.8rem; border: 1px solid #263653; border-radius: 22px;
               background: linear-gradient(110deg, rgba(17,30,52,.96), rgba(13,22,39,.86) 60%, rgba(30,38,64,.76)); margin-bottom: 1.2rem; }}
      .hero:before {{ content: ""; position:absolute; inset:0; opacity:.52; pointer-events:none;
        background-image: radial-gradient(1px 1px at 12% 22%, #fff 99%, transparent),
        radial-gradient(1px 1px at 34% 70%, #9be7ff 99%, transparent),
        radial-gradient(1.5px 1.5px at 67% 27%, #fff 99%, transparent),
        radial-gradient(1px 1px at 82% 63%, #fff 99%, transparent),
        radial-gradient(1px 1px at 94% 15%, #b8a5ff 99%, transparent),
        radial-gradient(1px 1px at 54% 10%, #fff 99%, transparent); }}
      .hero-grid {{ display:flex; align-items:center; justify-content:space-between; gap:1.2rem; position:relative; z-index:1; }}
      .hero h1 {{ font-size: clamp(2rem, 4vw, 3.25rem); line-height:1.03; margin:.45rem 0 .8rem; max-width:780px; }}
      .hero p {{ color:#B8C6E0; max-width:720px; margin:0; line-height:1.65; }}
      .orbit-art {{ width:170px; min-width:130px; height:150px; position:relative; }}
      .orbit-ring {{ position:absolute; width:145px; height:68px; border:1px solid rgba(108,229,232,.65); border-radius:50%; top:43px; left:10px; transform:rotate(-27deg); }}
      .orbit-ring.two {{ transform:rotate(52deg); border-color:rgba(184,165,255,.5); width:130px; left:18px; top:45px; }}
      .planet {{ position:absolute; width:58px; height:58px; border-radius:50%; left:53px; top:48px;
        background:radial-gradient(circle at 30% 28%, #F8D7A3, #B87C55 44%, #583F52 76%, #171E36); box-shadow:0 0 36px rgba(255,190,105,.17); }}
      .asteroid {{ position:absolute; width:17px; height:14px; border-radius:42% 58% 51% 49%; background:linear-gradient(135deg,#E2D1B4,#776B78); top:18px; left:125px; transform:rotate(32deg); box-shadow:0 0 15px #9c8a9a55; }}
      .metric-card {{ background:linear-gradient(145deg, {PANEL}, #0D1627); border:1px solid #22314B; border-radius:16px; padding:1rem 1.1rem; min-height:116px; }}
      .metric-label {{ color:{MUTED}; font-size:.74rem; text-transform:uppercase; letter-spacing:.11em; font-family:'DM Mono',monospace; }}
      .metric-value {{ color:{TEXT}; font-size:1.8rem; font-weight:800; letter-spacing:-.05em; margin:.32rem 0 .1rem; }}
      .metric-note {{ color:{MUTED}; font-size:.78rem; }}
      .section-head {{ display:flex; align-items:baseline; justify-content:space-between; margin:1.3rem 0 .55rem; gap:1rem; }}
      .section-head h2 {{ margin:0; font-size:1.3rem; }}
      .section-head span {{ color:{MUTED}; font-size:.8rem; }}
      .insight {{ background:linear-gradient(110deg,#102A37,#101B2F); border:1px solid #23505B; border-left:3px solid {CYAN}; border-radius:12px; padding:.85rem 1rem; color:#D7F6F5; line-height:1.55; }}
      .warning {{ background:#2B2024; border:1px solid #66434B; border-left:3px solid {CORAL}; border-radius:12px; padding:.85rem 1rem; color:#FFDCE0; line-height:1.55; }}
      .small-muted {{ color:{MUTED}; font-size:.83rem; }}
      .tag {{ display:inline-block; font-family:'DM Mono',monospace; font-size:.68rem; letter-spacing:.08em; padding:.3rem .55rem; border-radius:99px; border:1px solid #334562; color:#C7D5EE; }}
      .stTabs [data-baseweb="tab-list"] {{ gap:8px; border-bottom:1px solid #24334C; }}
      .stTabs [data-baseweb="tab"] {{ background:#10192B; border:1px solid #24334C; border-radius:9px 9px 0 0; padding:9px 15px; }}
      .stTabs [aria-selected="true"] {{ background:#1A3044 !important; border-color:#3B6877 !important; }}
      div[data-testid="stDataFrame"] {{ border:1px solid #273650; border-radius:12px; overflow:hidden; }}
      .stButton > button, .stDownloadButton > button {{ border-radius:10px; border:1px solid #3A6975; background:#173442; color:#DDFBFC; font-weight:700; }}
      .stButton > button:hover, .stDownloadButton > button:hover {{ border-color:{CYAN}; color:white; }}
      [data-testid="stMetric"] {{ background:{PANEL}; border:1px solid #22314B; padding:1rem; border-radius:14px; }}
      hr {{ border-color:#26334A; }}
      footer {{ visibility:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def style_plot(fig, height=390):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=TEXT, size=12),
        margin=dict(l=14, r=14, t=48, b=14),
        height=height,
        title=dict(font=dict(size=15, color=TEXT), x=0.02, xanchor="left"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        hoverlabel=dict(bgcolor="#15243A", bordercolor="#3B526F", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#223047", zerolinecolor="#2B3B54")
    fig.update_yaxes(showgrid=True, gridcolor="#223047", zerolinecolor="#2B3B54")
    return fig

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

def num_col(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")

def prepare_data(raw):
    df = raw.copy()
    if "pha" not in df.columns:
        return None, "The CSV must contain the NASA PHA target column named `pha`."
    df["pha"] = df["pha"].astype("string").str.strip().str.upper()
    df = df[df["pha"].isin(["Y", "N"])].copy()
    if df.empty:
        return None, "No rows with `pha` equal to Y or N were found."
    df["PHA_Label"] = df["pha"].map({"N": 0, "Y": 1}).astype(int)
    for c in ["a", "e", "q", "i", "om", "w", "ad", "n", "per_y", "moid", "H"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Circular encoding matches the project's selected main feature set.
    for angle in ["om", "w"]:
        if angle in df.columns:
            radians = np.deg2rad(df[angle])
            df[f"{angle}_sin"] = np.sin(radians)
            df[f"{angle}_cos"] = np.cos(radians)
    return df, None

MAIN_FEATURES = ["a", "e", "q", "i", "om_sin", "om_cos", "w_sin", "w_cos", "ad", "n", "per_y"]
FEATURE_LABELS = {
    "a": "Semi-major axis (AU)", "e": "Eccentricity", "q": "Perihelion distance (AU)",
    "i": "Inclination (°)", "om_sin": "Ascending node · sin", "om_cos": "Ascending node · cos",
    "w_sin": "Argument of perihelion · sin", "w_cos": "Argument of perihelion · cos",
    "ad": "Aphelion distance (AU)", "n": "Mean motion", "per_y": "Orbital period (years)",
}

@st.cache_data(show_spinner=False)
def cached_read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

@st.cache_resource(show_spinner=False)
def train_model(df_model):
    X = df_model[MAIN_FEATURES].copy()
    y = df_model["PHA_Label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, probs

def model_scores(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "PR-AUC": average_precision_score(y_true, probs),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }, pred

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="eyebrow">ORBITAL / MISSION CONTROL</div>', unsafe_allow_html=True)
    st.markdown("## Data console")
    st.markdown('<span class="tag">NEO · PHA CLASSIFICATION</span>', unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader(
        "Load asteroid catalogue",
        type=["csv"],
        help="Upload the NASA/JPL NEO CSV used in your notebook. It needs the `pha` column."
    )
    st.markdown("---")
    st.markdown("**Explore the catalogue**")
    st.caption("Filters update the charts and table immediately.")
    show_only = st.selectbox("PHA class", ["All objects", "Potentially hazardous (Y)", "Not classified as PHA (N)"])
    if uploaded is not None:
        st.caption("CSV loaded for this session.")
    st.markdown("---")
    st.markdown("**Reading the risk labels**")
    st.markdown(
        '<div class="small-muted">PHA = Potentially Hazardous Asteroid. '
        'This dashboard is an educational analysis tool, not an operational impact-warning system.</div>',
        unsafe_allow_html=True,
    )

# Load from upload or ask for the project CSV
if uploaded is None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">ASTEROID INTELLIGENCE LAB · PROJECT DASHBOARD</div>
              <h1>Mapping the objects<br>that move near Earth.</h1>
              <p>Explore orbital patterns, inspect the PHA class balance, and evaluate a machine-learning classifier through an interactive astronomy-themed workspace.</p>
              <div style="margin-top:1.1rem"><span class="tag">NASA/JPL NEO CATALOGUE</span>&nbsp; <span class="tag">INTERACTIVE ANALYTICS</span>&nbsp; <span class="tag">ML EVALUATION</span></div>
            </div>
            <div class="orbit-art"><div class="orbit-ring"></div><div class="orbit-ring two"></div><div class="planet"></div><div class="asteroid"></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("To open the dashboard, upload your asteroid catalogue CSV in the left sidebar. Use the same NASA/JPL NEO dataset from your notebook.")
    st.markdown("### What you can explore")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**01 · Catalogue overview**")
        st.write("Class balance, orbital distributions, and an interactive asteroid map of orbital characteristics.")
    with c2:
        st.markdown("**02 · Orbital explorer**")
        st.write("Filter by PHA label and compare semi-major axis, eccentricity, inclination, and perihelion distance.")
    with c3:
        st.markdown("**03 · Model laboratory**")
        st.write("Train the notebook's selected circular-feature HGB classifier and inspect PR-AUC, ROC-AUC, threshold metrics, and a confusion matrix.")
    st.stop()

try:
    raw_df = cached_read_csv(uploaded.getvalue())
except Exception as e:
    st.error(f"Could not read this CSV: {e}")
    st.stop()

df, error = prepare_data(raw_df)
if error:
    st.error(error)
    st.stop()

# Sidebar class filter is applied to exploration, not to model training.
filtered = df.copy()
if show_only == "Potentially hazardous (Y)":
    filtered = filtered[filtered["PHA_Label"] == 1]
elif show_only == "Not classified as PHA (N)":
    filtered = filtered[filtered["PHA_Label"] == 0]

n_all = len(df)
n_pha = int(df["PHA_Label"].sum())
n_non = n_all - n_pha
prevalence = n_pha / n_all if n_all else 0
pha_pct = 100 * prevalence
missing_target = int(raw_df["pha"].isna().sum()) if "pha" in raw_df else 0

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow">ASTEROID INTELLIGENCE LAB · LIVE DATA VIEW</div>
          <h1>Near-Earth Object<br>Intelligence Dashboard</h1>
          <p>From orbital mechanics to machine learning — inspect the catalogue, find patterns in known PHA labels, and understand what the classifier can and cannot tell us.</p>
          <div style="margin-top:1rem"><span class="tag">CATALOGUE LOADED</span>&nbsp; <span class="tag">{n_all:,} LABELLED OBJECTS</span>&nbsp; <span class="tag">FILTER: {show_only.upper()}</span></div>
        </div>
        <div class="orbit-art"><div class="orbit-ring"></div><div class="orbit-ring two"></div><div class="planet"></div><div class="asteroid"></div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI row
k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("Labelled objects", f"{n_all:,}", "Rows with a known PHA label")
with k2: metric_card("PHA objects", f"{n_pha:,}", f"{pha_pct:.2f}% of labelled catalogue")
with k3: metric_card("Non-PHA objects", f"{n_non:,}", f"{100-pha_pct:.2f}% of labelled catalogue")
with k4: metric_card("Target balance", f"1 : {n_non/max(n_pha,1):.1f}", "Non-PHA : PHA")

st.markdown("")
st.markdown(
    '<div class="insight"><b>How to read this:</b> PHA is a catalogue classification, not a prediction of an imminent collision. '
    'The machine-learning panel estimates how well orbital features separate the catalogue labels; it does not calculate impact probability.</div>',
    unsafe_allow_html=True,
)

tab_overview, tab_orbit, tab_model, tab_data = st.tabs(
    ["◈  Mission overview", "◎  Orbital explorer", "⌁  Model laboratory", "▤  Catalogue records"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown('<div class="section-head"><h2>Catalogue signal</h2><span>Label distribution & orbital context</span></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        class_counts = df["PHA_Label"].map({0: "Non-PHA (N)", 1: "PHA (Y)"}).value_counts().reindex(["Non-PHA (N)", "PHA (Y)"], fill_value=0).reset_index()
        class_counts.columns = ["Class", "Objects"]
        fig = px.pie(
            class_counts, names="Class", values="Objects", hole=0.72,
            color="Class", color_discrete_map={"Non-PHA (N)": CYAN, "PHA (Y)": AMBER},
            title="Known PHA labels",
        )
        fig.update_traces(textinfo="percent", textfont_size=13, marker=dict(line=dict(color=BG, width=3)))
        fig.add_annotation(text=f"<b>{pha_pct:.1f}%</b><br><span style='font-size:10px'>PHA</span>", x=0.5, y=0.5, showarrow=False, font=dict(size=23, color=TEXT))
        st.plotly_chart(style_plot(fig, 355), use_container_width=True)
    with right:
        if all(c in df.columns for c in ["a", "e"]):
            sample = df.dropna(subset=["a", "e"]).copy()
            if len(sample) > 7000:
                sample = sample.sample(7000, random_state=42)
            sample["PHA class"] = sample["PHA_Label"].map({0: "Non-PHA", 1: "PHA"})
            fig = px.scatter(
                sample, x="a", y="e", color="PHA class",
                color_discrete_map={"Non-PHA": CYAN, "PHA": AMBER},
                opacity=0.66, render_mode="webgl",
                hover_data=[c for c in ["spkid", "H", "moid", "i", "q"] if c in sample.columns],
                title="Orbital distribution · semi-major axis vs eccentricity",
                labels={"a": "Semi-major axis, a (AU)", "e": "Eccentricity"},
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(style_plot(fig, 355), use_container_width=True)
        else:
            st.warning("The orbital scatter needs columns `a` and `e`.")
    st.markdown('<div class="section-head"><h2>Quick science notes</h2><span>Generated from the loaded catalogue</span></div>', unsafe_allow_html=True)
    n_with_moid = int(df["moid"].notna().sum()) if "moid" in df else 0
    if "moid" in df:
        moid_near = int((df["moid"] <= 0.05).sum())
        moid_note = f"{moid_near:,} labelled objects have MOID ≤ 0.05 AU in the uploaded data."
    else:
        moid_note = "The `moid` column is not present, so the close-approach proxy is unavailable."
    a, b = st.columns(2)
    with a:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Orbital distance coverage</div><div class="metric-value">{n_with_moid:,}</div><div class="metric-note">Labelled rows with a usable MOID value</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Close-approach proxy</div><div class="metric-value">0.05 AU</div><div class="metric-note">{moid_note}</div></div>', unsafe_allow_html=True)
    if "H" in df and "moid" in df:
        rule_coverage = df[["H", "moid"]].notna().all(axis=1).sum()
        st.caption(f"Reference only: MOID and absolute magnitude H are used in common PHA screening criteria. This view does not replace the official NASA/JPL classification.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ORBITAL EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_orbit:
    st.markdown('<div class="section-head"><h2>Explore orbital geometry</h2><span>Change the axes and filters to inspect the catalogue</span></div>', unsafe_allow_html=True)
    available_axes = [c for c in ["a", "e", "q", "i", "ad", "n", "per_y", "moid", "H"] if c in filtered.columns and filtered[c].notna().any()]
    if len(available_axes) < 2:
        st.warning("The uploaded file does not contain enough usable orbital columns for this explorer.")
    else:
        controls = st.columns([1, 1, 1.2])
        with controls[0]:
            x_col = st.selectbox("Horizontal axis", available_axes, index=available_axes.index("a") if "a" in available_axes else 0, key="orbit_x")
        with controls[1]:
            y_default = "e" if "e" in available_axes else available_axes[min(1, len(available_axes)-1)]
            y_col = st.selectbox("Vertical axis", available_axes, index=available_axes.index(y_default), key="orbit_y")
        with controls[2]:
            max_points = st.select_slider("Points shown", options=[1000, 2500, 5000, 10000, 20000], value=5000)
        plot_df = filtered.dropna(subset=[x_col, y_col]).copy()
        if len(plot_df) > max_points:
            plot_df = plot_df.sample(max_points, random_state=42)
        plot_df["PHA class"] = plot_df["PHA_Label"].map({0: "Non-PHA", 1: "PHA"})
        hover_cols = [c for c in ["spkid", "H", "moid", "q", "i"] if c in plot_df.columns and c not in [x_col, y_col]]
        fig = px.scatter(
            plot_df, x=x_col, y=y_col, color="PHA class",
            color_discrete_map={"Non-PHA": CYAN, "PHA": AMBER},
            opacity=0.72, render_mode="webgl", hover_data=hover_cols,
            labels={x_col: FEATURE_LABELS.get(x_col, x_col), y_col: FEATURE_LABELS.get(y_col, y_col)},
            title=f"{FEATURE_LABELS.get(x_col, x_col)} vs {FEATURE_LABELS.get(y_col, y_col)}",
        )
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(style_plot(fig, 470), use_container_width=True)
        d1, d2 = st.columns(2)
        with d1:
            col = st.selectbox("Distribution feature", available_axes, index=available_axes.index("q") if "q" in available_axes else 0)
            dist_df = filtered[[col, "PHA_Label"]].dropna().copy()
            dist_df["PHA class"] = dist_df["PHA_Label"].map({0: "Non-PHA", 1: "PHA"})
            fig = px.histogram(dist_df, x=col, color="PHA class", marginal="box", barmode="overlay", opacity=.72,
                               color_discrete_map={"Non-PHA": CYAN, "PHA": AMBER}, nbins=45,
                               title=f"Distribution of {FEATURE_LABELS.get(col, col)}")
            st.plotly_chart(style_plot(fig, 370), use_container_width=True)
        with d2:
            if "i" in filtered and "q" in filtered:
                violin = filtered[["i", "q", "PHA_Label"]].dropna().copy()
                violin["PHA class"] = violin["PHA_Label"].map({0: "Non-PHA", 1: "PHA"})
                fig = px.box(violin, x="PHA class", y="i", color="PHA class", points=False,
                             color_discrete_map={"Non-PHA": CYAN, "PHA": AMBER}, title="Inclination by PHA class",
                             labels={"i": "Inclination (°)"})
                st.plotly_chart(style_plot(fig, 370), use_container_width=True)
            else:
                st.info("Add the `i` and `q` fields to explore inclination comparisons.")
        st.markdown('<div class="small-muted">Tip: each point represents one labelled catalogue object. Use hover to inspect available identifiers and orbital fields.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL LAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown('<div class="section-head"><h2>Model laboratory</h2><span>Reproducible hold-out evaluation · circular orbital features</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight">The main model uses 11 orbital features, including sine/cosine encodings for the angular variables. '
        'The 80/20 stratified split uses random state 42. MOID and H are deliberately excluded from this model because they can directly encode the conventional PHA definition.</div>',
        unsafe_allow_html=True,
    )
    required = [c for c in MAIN_FEATURES if c not in df.columns]
    if required:
        st.warning("Model evaluation needs these orbital columns: " + ", ".join(required) + ". The catalogue explorer still works with the available fields.")
    elif df["PHA_Label"].nunique() < 2 or df["PHA_Label"].value_counts().min() < 5:
        st.warning("There are not enough examples in both classes to train a reliable classifier.")
    else:
        if st.button("Run model evaluation", type="primary", use_container_width=False):
            st.session_state["run_pha_model"] = True
        if st.session_state.get("run_pha_model", False):
            model_df = df.dropna(subset=["PHA_Label"]).copy()
            with st.spinner("Training the classifier and scoring the held-out test set…"):
                try:
                    model, X_test, y_test, probs = train_model(model_df)
                    st.session_state["pha_model_results"] = {
                        "X_test": X_test, "y_test": y_test, "probs": probs,
                        "rows": len(model_df),
                    }
                except Exception as e:
                    st.error(f"Model run failed: {e}")
            res = st.session_state.get("pha_model_results")
            if res:
                y_test, probs = res["y_test"], res["probs"]
                threshold = st.slider("Classification threshold", 0.05, 0.80, 0.1612, 0.01,
                                      help="Lower thresholds flag more objects as PHA, usually increasing recall and false positives.")
                scores, pred = model_scores(y_test, probs, threshold)
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: metric_card("PR-AUC", f"{scores['PR-AUC']:.3f}", "Primary metric for imbalanced classes")
                with m2: metric_card("ROC-AUC", f"{scores['ROC-AUC']:.3f}", "Ranking discrimination")
                with m3: metric_card("Precision", f"{scores['Precision']:.3f}", "Share of flagged objects that are PHA")
                with m4: metric_card("Recall", f"{scores['Recall']:.3f}", "Share of known PHA objects detected")
                with m5: metric_card("F1 score", f"{scores['F1']:.3f}", "Precision–recall balance")
                left, right = st.columns(2, gap="large")
                with left:
                    p, r, _ = precision_recall_curve(y_test, probs)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name="Classifier", line=dict(color=CYAN, width=3), fill="tozeroy", fillcolor="rgba(108,229,232,.09)"))
                    fig.add_hline(y=float(y_test.mean()), line_dash="dash", line_color=AMBER,
                                  annotation_text=f"No-skill baseline {y_test.mean():.3f}")
                    fig.update_layout(title="Precision–recall curve", xaxis_title="Recall", yaxis_title="Precision")
                    st.plotly_chart(style_plot(fig, 390), use_container_width=True)
                with right:
                    fpr, tpr, _ = roc_curve(y_test, probs)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Classifier", line=dict(color=VIOLET, width=3), fill="tozeroy", fillcolor="rgba(184,165,255,.09)"))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(color="#60708B", dash="dash")))
                    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
                    st.plotly_chart(style_plot(fig, 390), use_container_width=True)
                cm = confusion_matrix(y_test, pred, labels=[0, 1])
                fig = px.imshow(
                    cm, text_auto=True, aspect="auto",
                    x=["Predicted non-PHA", "Predicted PHA"],
                    y=["Actual non-PHA", "Actual PHA"],
                    color_continuous_scale=[[0, "#10192B"], [.5, "#23606C"], [1, CYAN]],
                    title=f"Confusion matrix · threshold {threshold:.2f}",
                    labels=dict(x="Prediction", y="Actual", color="Objects"),
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(style_plot(fig, 360), use_container_width=True)
                tn, fp, fn, tp = cm.ravel()
                st.markdown(
                    f'<div class="insight"><b>Threshold interpretation:</b> At {threshold:.2f}, the test set contains '
                    f'{tp} true PHA detections, {fn} missed known PHA objects, {fp} false alarms, and {tn} correctly identified non-PHA objects. '
                    'A threshold is a trade-off, not a universal safety setting.</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download test-set predictions",
                    data=pd.DataFrame({
                        "actual_PHA": y_test.to_numpy(),
                        "predicted_PHA": pred,
                        "PHA_score": probs,
                    }).to_csv(index=False).encode("utf-8"),
                    file_name="pha_test_predictions.csv",
                    mime="text/csv",
                )
                st.caption("Results may vary slightly from the notebook if the uploaded catalogue version, missing values, or package versions differ. This classifier is for coursework analysis only.")
        else:
            st.markdown(
                '<div class="metric-card"><div class="metric-label">Selected model</div><div class="metric-value">HistGradientBoosting</div>'
                '<div class="metric-note">Circular orbital feature set · median imputation · 80/20 stratified hold-out</div></div>',
                unsafe_allow_html=True,
            )
            st.write("Press **Run model evaluation** to train the model on the uploaded catalogue and view interactive metrics. The app trains from the CSV; it does not load a pre-saved model.")
            st.markdown("**Features used**")
            st.code(", ".join(MAIN_FEATURES), language="text")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: RECORDS
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-head"><h2>Catalogue records</h2><span>Search, inspect, and export the filtered rows</span></div>', unsafe_allow_html=True)
    search_term = st.text_input("Search object identifier (SPK-ID) or designation", placeholder="e.g. 2000433 or asteroid designation")
    records = filtered.copy()
    if search_term:
        mask = pd.Series(False, index=records.index)
        for c in ["spkid", "full_name", "pdes", "name"]:
            if c in records.columns:
                mask |= records[c].astype(str).str.contains(search_term, case=False, na=False)
        records = records[mask]
    display_cols = [c for c in ["spkid", "full_name", "pdes", "pha", "a", "e", "q", "i", "moid", "H"] if c in records.columns]
    st.caption(f"Showing {min(len(records), 1000):,} of {len(records):,} matching rows.")
    st.dataframe(records[display_cols].head(1000), use_container_width=True, hide_index=True)
    st.download_button(
        "Export filtered catalogue as CSV",
        data=records.to_csv(index=False).encode("utf-8"),
        file_name="pha_filtered_catalogue.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    '<div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;color:#7183A5;font-size:.75rem">'
    '<span>ORBITAL · Near-Earth Object Intelligence</span>'
    '<span>Academic prototype · Not for operational hazard assessment</span></div>',
    unsafe_allow_html=True,
)
