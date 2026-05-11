"""
HUPA-UCM Diabetes CGM Dashboard
================================
Streamlit dashboard for the HUPA-UCM Continuous Glucose Monitoring dataset.
Dataset: https://www.sciencedirect.com/science/article/pii/S2352340924005262

Features: CGM glucose, insulin doses, meals (carbs), steps, calories,
          heart rate, sleep quality — 25 T1DM patients, 5-min intervals.

Run:
    pip install streamlit pandas numpy plotly scipy scikit-learn
    streamlit run diabetes_cgm_dashboard.py

If you have the real CSV files, place them in a folder and update DATA_DIR below.
The app auto-generates synthetic demo data if no files are found.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings, os, datetime

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = "./data"          # folder containing patient CSV files (optional)
N_PATIENTS = 25
SEED = 42
np.random.seed(SEED)

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HUPA-UCM · Diabetes CGM Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

  /* Main background */
  .stApp { background-color: #0f1117; color: #e2e8f0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2d3748;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 16px !important;
  }
  [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 12px !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 28px !important;
    color: #e2e8f0 !important;
  }

  /* Section headers */
  .section-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 6px;
  }

  /* Risk badge */
  .badge-low    { background:#065f46; color:#6ee7b7; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
  .badge-target { background:#1e40af; color:#93c5fd; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
  .badge-high   { background:#7f1d1d; color:#fca5a5; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { background: #161b27; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 6px; }
  .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #1e293b; color: #e2e8f0; }

  /* Divider */
  hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)

# ── Data generation / loading ─────────────────────────────────────────────────
@st.cache_data
def generate_patient_data(patient_id: int, days: int = 14) -> pd.DataFrame:
    """Generate realistic synthetic CGM patient data at 5-min resolution."""
    rng = np.random.default_rng(patient_id * 17)
    n = days * 24 * 12  # 5-min intervals

    timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")

    # Glucose: baseline + circadian + meal spikes + noise
    t = np.arange(n)
    base = rng.uniform(100, 140)
    circadian = 15 * np.sin(2 * np.pi * t / (24 * 12) - np.pi / 2)
    noise = rng.normal(0, 8, n)

    # Meal spikes (3 meals/day)
    glucose = base + circadian + noise
    for day in range(days):
        for meal_h in [7, 12, 19]:
            idx = day * 24 * 12 + meal_h * 12
            spike = rng.uniform(40, 90)
            decay = np.exp(-np.arange(36) / 10)
            end = min(idx + 36, n)
            glucose[idx:end] += spike * decay[:end - idx]

    glucose = np.clip(glucose, 40, 400)

    # Insulin boluses aligned with meals
    insulin = np.zeros(n)
    for day in range(days):
        for meal_h in [7, 12, 19]:
            idx = day * 24 * 12 + meal_h * 12
            if idx < n:
                insulin[idx] = rng.uniform(2, 8)
        # Basal
        basal_idx = rng.choice(n, size=4, replace=False)
        insulin[basal_idx] = rng.uniform(0.5, 2, 4)

    # Carbs
    carbs = np.zeros(n)
    for day in range(days):
        for meal_h in [7, 12, 19]:
            idx = day * 24 * 12 + meal_h * 12
            if idx < n:
                carbs[idx] = rng.uniform(20, 80)

    # Steps (active during day)
    hour_of_day = (t // 12) % 24
    step_base = np.where((hour_of_day >= 8) & (hour_of_day <= 20), 20, 0)
    steps = np.clip(rng.poisson(step_base, n), 0, 200)

    # Heart rate
    hr_base = rng.uniform(62, 78)
    hr = hr_base + 0.02 * steps + rng.normal(0, 3, n)
    hr = np.clip(hr, 45, 160)

    # Calories (~same envelope as steps)
    calories = steps * 0.05 + rng.uniform(0.8, 1.2, n)

    # Sleep (flag: 1 = sleeping, 22:00-07:00)
    sleep = ((hour_of_day >= 22) | (hour_of_day < 7)).astype(int)
    sleep_quality = np.where(sleep == 1, rng.choice([1, 2, 3], n, p=[0.1, 0.3, 0.6]), 0)

    return pd.DataFrame({
        "timestamp": timestamps,
        "patient_id": patient_id,
        "glucose": np.round(glucose, 1),
        "insulin": np.round(insulin, 2),
        "carbs": np.round(carbs, 1),
        "steps": steps.astype(int),
        "heart_rate": np.round(hr, 1),
        "calories": np.round(calories, 2),
        "sleep": sleep,
        "sleep_quality": sleep_quality,
    })


@st.cache_data
def load_all_patients():
    """Load from files if DATA_DIR exists, else generate synthetic data."""
    patients = []

    if os.path.isdir(DATA_DIR):
        files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
        for i, fname in enumerate(files[:N_PATIENTS]):
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, fname))
                df["patient_id"] = i + 1
                patients.append(df)
            except Exception:
                patients.append(generate_patient_data(i + 1))
    else:
        for pid in range(1, N_PATIENTS + 1):
            patients.append(generate_patient_data(pid))

    return pd.concat(patients, ignore_index=True)


@st.cache_data
def compute_patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-patient statistics."""
    rows = []
    for pid, grp in df.groupby("patient_id"):
        g = grp["glucose"]
        tir   = ((g >= 70) & (g <= 180)).mean() * 100
        tbr   = (g < 70).mean() * 100
        tar   = (g > 180).mean() * 100
        cv    = g.std() / g.mean() * 100
        rows.append({
            "patient_id": pid,
            "mean_glucose": round(g.mean(), 1),
            "std_glucose": round(g.std(), 1),
            "tir": round(tir, 1),
            "tbr": round(tbr, 1),
            "tar": round(tar, 1),
            "cv": round(cv, 1),
            "total_insulin": round(grp["insulin"].sum(), 1),
            "total_carbs": round(grp["carbs"].sum(), 1),
            "mean_hr": round(grp["heart_rate"].mean(), 1),
            "mean_steps": round(grp["steps"].mean(), 1),
            "gmi": round(3.31 + 0.02392 * g.mean(), 2),   # GMI formula
        })
    return pd.DataFrame(rows)


# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#161b27",
    plot_bgcolor="#0f1117",
    font=dict(family="DM Mono, monospace", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=30, b=40),
)

COLORS = {
    "glucose": "#38bdf8",
    "insulin": "#818cf8",
    "carbs": "#34d399",
    "hr": "#f472b6",
    "steps": "#fb923c",
    "hypo": "#ef4444",
    "target": "#22c55e",
    "hyper": "#f59e0b",
}

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading patient data..."):
    df_all = load_all_patients()
    df_summary = compute_patient_summary(df_all)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩸 HUPA-UCM CGM")
    st.markdown("**Diabetes · T1DM Cohort**")
    st.markdown("---")

    selected_patient = st.selectbox(
        "Select patient",
        options=list(range(1, N_PATIENTS + 1)),
        format_func=lambda x: f"Patient {x:02d}",
    )

    patient_df = df_all[df_all["patient_id"] == selected_patient].copy()
    patient_df["date"] = patient_df["timestamp"].dt.date
    available_dates = sorted(patient_df["date"].unique())

    selected_date = st.selectbox(
        "Select day",
        options=available_dates,
        format_func=lambda d: d.strftime("%b %d, %Y"),
        index=min(3, len(available_dates) - 1),
    )

    day_df = patient_df[patient_df["date"] == selected_date].copy()

    st.markdown("---")
    st.markdown("**Glucose targets (mg/dL)**")
    hypo_thresh  = st.slider("Hypoglycemia below", 50, 90, 70)
    hyper_thresh = st.slider("Hyperglycemia above", 140, 300, 180)

    st.markdown("---")
    st.caption("Dataset: HUPA-UCM · 25 patients · T1DM\nDOI: 10.1016/j.dib.2024.110526")

# ── Header ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("# 🩸 Diabetes CGM Analytics Dashboard")
    st.markdown(
        "**HUPA-UCM dataset** · Continuous glucose monitoring · "
        f"25 T1DM patients · 5-min resolution · Viewing: **Patient {selected_patient:02d}**"
    )
with c2:
    p_summary = df_summary[df_summary["patient_id"] == selected_patient].iloc[0]
    tir = p_summary["tir"]
    badge_cls = "badge-target" if tir >= 70 else "badge-high" if tir < 50 else "badge-low"
    badge_txt = "Good control" if tir >= 70 else "Poor control" if tir < 50 else "Moderate"
    st.markdown(f"<br><span class='{badge_cls}'>{badge_txt}</span>", unsafe_allow_html=True)
    st.caption(f"TIR: {tir}% · GMI: {p_summary['gmi']}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Daily CGM",
    "📊 Patient Overview",
    "🏥 Cohort Analytics",
    "🤖 ML Risk Model",
    "📋 Data Explorer",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Daily CGM
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"<div class='section-title'>Daily glucose trace — {selected_date.strftime('%B %d, %Y')}</div>",
                unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    g_day = day_df["glucose"]
    k1.metric("Mean glucose", f"{g_day.mean():.0f} mg/dL")
    k2.metric("Std deviation", f"{g_day.std():.0f} mg/dL")
    k3.metric("Time in range", f"{((g_day>=70)&(g_day<=180)).mean()*100:.0f}%")
    k4.metric("Hypo events", int((g_day < hypo_thresh).sum()))
    k5.metric("Hyper events", int((g_day > hyper_thresh).sum()))

    # CGM trace + insulin + carbs
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.04,
        subplot_titles=["Glucose (mg/dL)", "Insulin doses (U)", "Carbohydrates (g)"],
    )

    t = day_df["timestamp"]

    # Hypo / hyper bands
    fig.add_hrect(y0=0, y1=hypo_thresh, fillcolor="#ef444420", line_width=0, row=1, col=1)
    fig.add_hrect(y0=hyper_thresh, y1=400, fillcolor="#f59e0b20", line_width=0, row=1, col=1)
    fig.add_hline(y=hypo_thresh, line_dash="dot", line_color="#ef4444", line_width=1, row=1, col=1)
    fig.add_hline(y=hyper_thresh, line_dash="dot", line_color="#f59e0b", line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=day_df["glucose"],
        mode="lines", name="CGM glucose",
        line=dict(color=COLORS["glucose"], width=2),
        fill="tozeroy", fillcolor="#38bdf808",
    ), row=1, col=1)

    # Meal marker overlay
    meals = day_df[day_df["carbs"] > 0]
    fig.add_trace(go.Scatter(
        x=meals["timestamp"], y=meals["glucose"],
        mode="markers", name="Meal",
        marker=dict(symbol="triangle-up", color=COLORS["carbs"], size=10),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=t, y=day_df["insulin"],
        name="Insulin", marker_color=COLORS["insulin"],
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=t, y=day_df["carbs"],
        name="Carbs", marker_color=COLORS["carbs"],
    ), row=3, col=1)

    fig.update_layout(
        height=560, showlegend=True,
        legend=dict(orientation="h", y=1.04),
        **PLOTLY_THEME,
    )
    fig.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)

    # Heart rate + steps
    st.markdown("<div class='section-title'>Activity & vitals</div>", unsafe_allow_html=True)

    fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Heart Rate (bpm)", "Steps"])
    fig2.add_trace(go.Scatter(
        x=t, y=day_df["heart_rate"], mode="lines",
        line=dict(color=COLORS["hr"], width=1.5), name="Heart rate",
    ), row=1, col=1)
    fig2.add_trace(go.Bar(
        x=t, y=day_df["steps"], marker_color=COLORS["steps"], name="Steps",
    ), row=1, col=2)
    fig2.update_layout(height=260, showlegend=False, **PLOTLY_THEME)
    fig2.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Patient Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Full recording — glucose & activity overview</div>",
                unsafe_allow_html=True)

    # Multi-day glucose heatmap by hour
    p_df = patient_df.copy()
    p_df["hour"] = p_df["timestamp"].dt.hour
    p_df["day_num"] = (p_df["timestamp"].dt.date - p_df["timestamp"].dt.date.min()).apply(lambda x: x.days)
    hm = p_df.groupby(["day_num", "hour"])["glucose"].mean().unstack(fill_value=np.nan)

    fig_hm = go.Figure(go.Heatmap(
        z=hm.values,
        x=[f"{h:02d}:00" for h in hm.columns],
        y=[f"Day {d+1}" for d in hm.index],
        colorscale=[
            [0.0, "#1e3a5f"], [0.25, "#38bdf8"],
            [0.6, "#22c55e"], [0.8, "#f59e0b"], [1.0, "#ef4444"]
        ],
        colorbar=dict(title="Glucose<br>(mg/dL)"),
        zmin=60, zmax=280,
    ))
    fig_hm.update_layout(
        title="Glucose heatmap by hour of day",
        height=420, **PLOTLY_THEME,
        xaxis=dict(title="Hour of day"),
        yaxis=dict(title="Study day", autorange="reversed"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # TIR donut
        p_s = df_summary[df_summary["patient_id"] == selected_patient].iloc[0]
        fig_tir = go.Figure(go.Pie(
            labels=["Time in range (70–180)", "Hypoglycemia (<70)", "Hyperglycemia (>180)"],
            values=[p_s["tir"], p_s["tbr"], p_s["tar"]],
            hole=0.62,
            marker_colors=["#22c55e", "#ef4444", "#f59e0b"],
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_tir.add_annotation(
            text=f"TIR<br><b>{p_s['tir']}%</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#e2e8f0"),
        )
        fig_tir.update_layout(
            title="Time in range breakdown",
            height=320, showlegend=False, **PLOTLY_THEME,
        )
        st.plotly_chart(fig_tir, use_container_width=True)

    with col_b:
        # Glucose distribution
        fig_dist = go.Figure()
        fig_dist.add_vrect(x0=0, x1=hypo_thresh, fillcolor="#ef444415", line_width=0)
        fig_dist.add_vrect(x0=hyper_thresh, x1=400, fillcolor="#f59e0b15", line_width=0)
        fig_dist.add_trace(go.Histogram(
            x=patient_df["glucose"],
            nbinsx=60,
            marker_color=COLORS["glucose"],
            opacity=0.8,
            name="Glucose",
        ))
        fig_dist.update_layout(
            title="Glucose distribution (full recording)",
            height=320,
            xaxis_title="Glucose (mg/dL)",
            yaxis_title="Count",
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Glucose vs carbs scatter
    st.markdown("<div class='section-title'>Post-meal glucose response</div>", unsafe_allow_html=True)
    meal_events = p_df[p_df["carbs"] > 0].copy()
    meal_events["glucose_30min_later"] = np.nan
    for idx in meal_events.index:
        future = p_df.loc[idx:idx+6, "glucose"]
        if len(future) >= 4:
            meal_events.at[idx, "glucose_30min_later"] = future.max()

    meal_events = meal_events.dropna(subset=["glucose_30min_later"])
    if len(meal_events) > 0:
        fig_meal = px.scatter(
            meal_events, x="carbs", y="glucose_30min_later",
            trendline="ols", color="hour",
            color_continuous_scale="teal",
            labels={"carbs": "Carbohydrates (g)", "glucose_30min_later": "Peak glucose 30 min post-meal (mg/dL)", "hour": "Hour"},
            title="Carbohydrate load vs peak post-meal glucose",
            template="plotly_dark",
        )
        fig_meal.update_layout(height=320, **PLOTLY_THEME)
        st.plotly_chart(fig_meal, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cohort Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Cohort summary · all 25 patients</div>",
                unsafe_allow_html=True)

    # Summary table metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cohort mean glucose", f"{df_summary['mean_glucose'].mean():.0f} mg/dL")
    m2.metric("Mean TIR", f"{df_summary['tir'].mean():.1f}%")
    m3.metric("% patients TIR ≥ 70%", f"{(df_summary['tir'] >= 70).mean()*100:.0f}%")
    m4.metric("Mean GMI", f"{df_summary['gmi'].mean():.2f}")

    col1, col2 = st.columns(2)

    with col1:
        # TIR bar chart all patients
        fig_tir_all = go.Figure()
        colors_bar = ["#22c55e" if t >= 70 else "#f59e0b" if t >= 50 else "#ef4444"
                      for t in df_summary["tir"]]
        fig_tir_all.add_trace(go.Bar(
            x=[f"P{p:02d}" for p in df_summary["patient_id"]],
            y=df_summary["tir"],
            marker_color=colors_bar,
            text=df_summary["tir"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig_tir_all.add_hline(y=70, line_dash="dash", line_color="#94a3b8",
                               annotation_text="Target 70%")
        fig_tir_all.update_layout(
            title="Time in range per patient",
            height=320, xaxis_title="Patient", yaxis_title="TIR (%)",
            yaxis_range=[0, 105],
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_tir_all, use_container_width=True)

    with col2:
        # CV vs mean glucose scatter
        fig_cv = px.scatter(
            df_summary,
            x="mean_glucose", y="cv",
            color="tir",
            color_continuous_scale="RdYlGn",
            size="total_insulin",
            hover_name="patient_id",
            hover_data={"mean_glucose": True, "cv": True, "tir": True},
            labels={"mean_glucose": "Mean glucose (mg/dL)", "cv": "Coefficient of variation (%)", "tir": "TIR (%)"},
            title="Glycemic variability vs mean glucose",
            template="plotly_dark",
        )
        fig_cv.add_vline(x=154, line_dash="dot", line_color="#94a3b8",
                          annotation_text="GMI 7%")
        fig_cv.update_layout(height=320, **PLOTLY_THEME)
        st.plotly_chart(fig_cv, use_container_width=True)

    # Parallel coords
    st.markdown("<div class='section-title'>Multi-dimensional patient profile</div>",
                unsafe_allow_html=True)

    fig_par = px.parallel_coordinates(
        df_summary,
        dimensions=["mean_glucose", "tir", "tbr", "tar", "cv", "mean_hr", "mean_steps"],
        color="tir",
        color_continuous_scale=px.colors.sequential.Teal,
        labels={
            "mean_glucose": "Mean BG", "tir": "TIR%", "tbr": "TBR%",
            "tar": "TAR%", "cv": "CV%", "mean_hr": "HR", "mean_steps": "Steps",
        },
        template="plotly_dark",
        title="Patient profile parallel coordinates — color = TIR%",
    )
    fig_par.update_layout(height=380, **PLOTLY_THEME)
    st.plotly_chart(fig_par, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-title'>Feature correlations</div>", unsafe_allow_html=True)
    corr_cols = ["mean_glucose", "std_glucose", "tir", "tbr", "tar",
                 "cv", "total_insulin", "total_carbs", "mean_hr", "mean_steps", "gmi"]
    corr = df_summary[corr_cols].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr_cols, y=corr_cols,
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="r"),
    ))
    fig_corr.update_layout(
        title="Pearson correlation matrix",
        height=420, **PLOTLY_THEME,
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML Risk Model
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Glycemic control risk classifier</div>",
                unsafe_allow_html=True)
    st.caption("Predicts whether a patient has **Poor control** (TIR < 50%) or **Good/Moderate control** using Random Forest.")

    @st.cache_data
    def train_model(summary_df):
        features = ["mean_glucose", "std_glucose", "cv", "total_insulin",
                    "total_carbs", "mean_hr", "mean_steps"]
        X = summary_df[features].values
        y = (summary_df["tir"] < 50).astype(int).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
        rf.fit(X_scaled, y)
        scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")
        return rf, scaler, features, scores

    rf_model, scaler, feat_names, cv_scores = train_model(df_summary)

    col_ml1, col_ml2 = st.columns([1, 1])

    with col_ml1:
        st.markdown("**Cross-validation accuracy**")
        cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(5)],
                               "Accuracy": cv_scores})
        fig_cv_scores = go.Figure(go.Bar(
            x=cv_df["Fold"], y=cv_df["Accuracy"],
            marker_color=COLORS["glucose"],
            text=[f"{s:.0%}" for s in cv_scores],
            textposition="outside",
        ))
        fig_cv_scores.add_hline(y=cv_scores.mean(), line_dash="dash",
                                  line_color="#94a3b8",
                                  annotation_text=f"Mean {cv_scores.mean():.0%}")
        fig_cv_scores.update_layout(
            height=260, yaxis_range=[0, 1.15],
            yaxis_tickformat=".0%",
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_cv_scores, use_container_width=True)

    with col_ml2:
        # Feature importance
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})\
                  .sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker_color=COLORS["insulin"],
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig_fi.update_layout(
            title="Feature importance",
            height=280,
            xaxis_title="Importance",
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Live predictor
    st.markdown("<div class='section-title'>Real-time risk predictor</div>", unsafe_allow_html=True)
    st.caption("Adjust sliders to estimate risk for a new patient profile.")

    sp1, sp2, sp3, sp4 = st.columns(4)
    inp_glucose  = sp1.slider("Mean glucose", 80, 250, 130)
    inp_std      = sp2.slider("Std deviation", 5, 80, 30)
    inp_cv       = sp3.slider("CV (%)", 5, 60, 25)
    inp_insulin  = sp4.slider("Total insulin (U)", 50, 1000, 300)

    sp5, sp6, sp7, _ = st.columns(4)
    inp_carbs    = sp5.slider("Total carbs (g)", 500, 5000, 2000)
    inp_hr       = sp6.slider("Mean HR (bpm)", 50, 100, 68)
    inp_steps    = sp7.slider("Mean steps/5min", 0, 100, 15)

    X_new = scaler.transform([[inp_glucose, inp_std, inp_cv,
                                inp_insulin, inp_carbs, inp_hr, inp_steps]])
    prob = rf_model.predict_proba(X_new)[0][1]

    r1, r2, r3 = st.columns(3)
    r1.metric("Poor control probability", f"{prob*100:.1f}%")
    r2.metric("Predicted class",
              "⚠️ Poor control" if prob > 0.5 else "✅ Adequate control")
    r3.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "family": "DM Mono"}},
        title={"text": "Poor glycemic control risk", "font": {"color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": "#ef4444" if prob > 0.5 else "#22c55e"},
            "bgcolor": "#1e293b",
            "steps": [
                {"range": [0, 33], "color": "#14532d30"},
                {"range": [33, 66], "color": "#78350f30"},
                {"range": [66, 100], "color": "#7f1d1d30"},
            ],
            "threshold": {
                "line": {"color": "#94a3b8", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig_gauge.update_layout(
        height=280,
        paper_bgcolor="#161b27",
        font=dict(color="#94a3b8"),
        margin=dict(l=30, r=30, t=40, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>Raw data explorer</div>",
                unsafe_allow_html=True)

    view_scope = st.radio("View", ["Patient summary", "Raw CGM (selected patient)"], horizontal=True)

    if view_scope == "Patient summary":
        st.dataframe(
            df_summary.style.background_gradient(
                subset=["tir"], cmap="RdYlGn", vmin=0, vmax=100
            ).background_gradient(
                subset=["tbr"], cmap="Reds_r", vmin=0, vmax=30
            ).format(precision=1),
            use_container_width=True,
            height=500,
        )
    else:
        n_rows = st.slider("Rows to display", 50, 2000, 288)
        raw_view = patient_df.drop(columns=["date"]).head(n_rows)
        st.dataframe(
            raw_view.style.background_gradient(
                subset=["glucose"], cmap="RdYlGn_r", vmin=60, vmax=280
            ).format(precision=2),
            use_container_width=True,
            height=480,
        )

    st.download_button(
        "⬇ Download patient summary CSV",
        data=df_summary.to_csv(index=False),
        file_name="hupa_ucm_patient_summary.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "HUPA-UCM Dataset · Univ. de Alcalá · "
    "DOI: 10.1016/j.dib.2024.110526 · "
    "Dashboard built with Streamlit + Plotly"
)
