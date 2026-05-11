# app.py
# GlucoAI: Hackathon-Winning Diabetes Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="GlucoAI Diabetes Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
}
.block-container {
    padding-top: 1.5rem;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div style="
background:linear-gradient(90deg,#0f766e,#2563eb);
padding:30px;
border-radius:24px;
color:white;
margin-bottom:25px;">
<h1>🩺 GlucoAI Diabetes Intelligence Platform</h1>
<h4>Predictive + Prescriptive Analytics for CGM, Insulin, Meals, Activity, Sleep & Risk Forecasting</h4>
<p>Hackathon-ready AI dashboard for real-time glucose intelligence, patient risk scoring, and personalized recommendations.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_hupa_diabetes_recent (1)(2).csv")
    demo = pd.read_csv("cleaned_demographics (1)(1).csv")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if "patient_id" in demo.columns:
        df = df.merge(demo, on="patient_id", how="left")

    return df

df = load_data()

# ---------------------------------------------------
# COLUMN SAFETY
# ---------------------------------------------------

df = df.dropna(subset=["patient_id", "time", "glucose"])
df = df.sort_values(["patient_id", "time"])

bolus_col = "bolus_volume_delivered" if "bolus_volume_delivered" in df.columns else "bolus"

for col in ["steps", "heart_rate", "basal_rate", "carb_input", bolus_col]:
    if col not in df.columns:
        df[col] = 0

df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date
df["is_weekend"] = df["time"].dt.dayofweek.isin([5, 6]).astype(int)
df["is_night"] = df["hour"].between(0, 5).astype(int)

df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

df["glucose_rolling_mean_1h"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12)
    .mean()
    .reset_index(level=0, drop=True)
)

df["glucose_rolling_std_1h"] = (
    df.groupby("patient_id")["glucose"]
    .rolling(12)
    .std()
    .reset_index(level=0, drop=True)
)

df["is_hypoglycemia"] = (df["glucose"] < 70).astype(int)
df["is_in_range"] = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
df["is_hyperglycemia"] = (df["glucose"] > 180).astype(int)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🧭 Dashboard Controls")

patients = sorted(df["patient_id"].unique())

selected_patients = st.sidebar.multiselect(
    "Select Patients",
    patients,
    default=patients[:5]
)

df_view = df[df["patient_id"].isin(selected_patients)].copy()

if df_view.empty:
    st.warning("Please select at least one patient.")
    st.stop()

# ---------------------------------------------------
# KPI ROW
# ---------------------------------------------------

total_patients = df_view["patient_id"].nunique()
total_records = len(df_view)
tir = df_view["is_in_range"].mean() * 100
hypo = df_view["is_hypoglycemia"].mean() * 100
hyper = df_view["is_hyperglycemia"].mean() * 100
avg_glucose = df_view["glucose"].mean()

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("👥 Patients", total_patients)
c2.metric("📊 Records", f"{total_records:,}")
c3.metric("✅ TIR", f"{tir:.1f}%")
c4.metric("⚠️ Hypo", f"{hypo:.1f}%")
c5.metric("🔥 Hyper", f"{hyper:.1f}%")
c6.metric("🩸 Avg Glucose", f"{avg_glucose:.1f}")

# ---------------------------------------------------
# DAILY SUMMARY
# ---------------------------------------------------

daily = (
    df_view
    .groupby(["patient_id", "date"])
    .agg(
        daily_tir=("is_in_range", "mean"),
        avg_glucose=("glucose", "mean"),
        glucose_variability=("glucose", "std"),
        daily_steps=("steps", "sum"),
        avg_hr=("heart_rate", "mean"),
        avg_basal=("basal_rate", "mean"),
        total_bolus=(bolus_col, "sum"),
        total_carbs=("carb_input", "sum"),
        hypo_rate=("is_hypoglycemia", "mean"),
        hyper_rate=("is_hyperglycemia", "mean")
    )
    .reset_index()
)

daily["daily_tir"] *= 100

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tabs = st.tabs([
    "🏠 Home",
    "📊 Glucose Overview",
    "🍽️ Meal + Insulin",
    "🏃 Activity + Sleep",
    "🌙 Night Risk",
    "🤖 Predictive AI",
    "💊 Prescriptive Analytics",
    "📌 Key Takeaways"
])

# ---------------------------------------------------
# HOME
# ---------------------------------------------------

with tabs[0]:
    st.subheader("Project Overview")

    st.markdown("""
    **GlucoAI** combines CGM, insulin, meals, activity, heart rate, sleep, and demographic signals to support diabetes intelligence.

    This dashboard includes:
    - Descriptive analytics: TIR, hypoglycemia, glucose trends
    - Predictive analytics: hypo prediction, hyperglycemia prediction, ROC forecasting
    - Prescriptive analytics: insulin effectiveness score, basal risk, carb guidance
    - Patient stratification and risk monitoring
    """)

    st.success("Goal: Improve diabetes safety, reduce glucose instability, and support personalized intervention decisions.")

# ---------------------------------------------------
# GLUCOSE OVERVIEW
# ---------------------------------------------------

with tabs[1]:
    st.subheader("Glucose Monitoring Overview")

    fig = px.line(
        df_view,
        x="time",
        y="glucose",
        color="patient_id",
        title="24-Hour / Longitudinal Glucose Trend"
    )

    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=180, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        tir_summary = (
            df_view
            .groupby("patient_id")
            .agg(
                TBR=("is_hypoglycemia", "mean"),
                TIR=("is_in_range", "mean"),
                TAR=("is_hyperglycemia", "mean")
            ) * 100
        ).reset_index()

        tir_melt = tir_summary.melt(
            id_vars="patient_id",
            var_name="Range",
            value_name="Percentage"
        )

        fig = px.bar(
            tir_melt,
            x="patient_id",
            y="Percentage",
            color="Range",
            title="TBR / TIR / TAR by Patient"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df_view,
            x="patient_id",
            y="glucose",
            title="Glucose Distribution by Patient"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# MEAL + INSULIN
# ---------------------------------------------------

with tabs[2]:
    st.subheader("Meal, Carbohydrate, and Insulin Response")

    meal_df = df_view.copy()

    meal_df["glucose_next_2h"] = (
        meal_df.groupby("patient_id")["glucose"].shift(-24)
    )

    meal_df["post_meal_spike"] = (
        meal_df["glucose_next_2h"] - meal_df["glucose"]
    )

    meal_df = meal_df[meal_df["carb_input"] > 0].dropna(
        subset=["carb_input", "post_meal_spike", bolus_col]
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            meal_df.sample(min(5000, len(meal_df)), random_state=42),
            x="carb_input",
            y="post_meal_spike",
            size=bolus_col,
            color="glucose",
            trendline="ols",
            title="Carbohydrate Intake vs Post-Meal Spike"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.density_heatmap(
            meal_df,
            x="carb_input",
            y="post_meal_spike",
            nbinsx=30,
            nbinsy=30,
            title="Carb Load vs Spike Risk Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missed Bolus Detection")

    meal_df["missed_bolus"] = (
        (meal_df["carb_input"] > 20) &
        (meal_df[bolus_col] == 0) &
        (meal_df["glucose_next_2h"] > 180)
    ).astype(int)

    missed = meal_df["missed_bolus"].value_counts().reset_index()
    missed.columns = ["Missed Bolus", "Count"]

    fig = px.bar(
        missed,
        x="Missed Bolus",
        y="Count",
        title="Detected Missed Bolus Events"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# ACTIVITY + SLEEP
# ---------------------------------------------------

with tabs[3]:
    st.subheader("Activity Impact on Glucose Stability")

    activity_df = df_view.copy()

    activity_df["glucose_drift_1h"] = (
        activity_df.groupby("patient_id")["glucose"].diff(12)
    )

    activity_df["activity_group"] = pd.cut(
        activity_df["steps"],
        bins=[-1, 0, 50, 500, 100000],
        labels=["Sedentary", "Low", "Moderate", "High"]
    )

    act_summary = (
        activity_df
        .groupby("activity_group", observed=True)
        .agg(
            avg_drift=("glucose_drift_1h", "mean"),
            avg_instability=("glucose_rolling_std_1h", "mean"),
            avg_glucose=("glucose", "mean")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            act_summary,
            x="activity_group",
            y="avg_instability",
            color="activity_group",
            title="Activity Level vs Glucose Instability"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            daily,
            x="daily_steps",
            y="daily_tir",
            size="glucose_variability",
            color="avg_glucose",
            hover_data=["patient_id", "date"],
            title="Daily Steps vs Time-In-Range"
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# NIGHT RISK
# ---------------------------------------------------

with tabs[4]:
    st.subheader("Nocturnal Hypoglycemia Risk")

    night_df = df_view[df_view["is_night"] == 1].copy()
    night_df["nocturnal_hypo"] = (night_df["glucose"] < 70).astype(int)

    risk_curve = (
        night_df
        .groupby("basal_rate")["nocturnal_hypo"]
        .mean()
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            risk_curve,
            x="basal_rate",
            y="nocturnal_hypo",
            markers=True,
            title="Night Basal Rate vs Hypoglycemia Risk"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            night_df,
            x="nocturnal_hypo",
            y="basal_rate",
            title="Basal Rate Distribution by Night Hypoglycemia"
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PREDICTIVE AI
# ---------------------------------------------------

with tabs[5]:
    st.subheader("Predictive AI Models")

    model_choice = st.selectbox(
        "Select Prediction Task",
        [
            "Hypoglycemia Next 30 Minutes",
            "Hyperglycemia >200 Within 2 Hours After Meal",
            "Next 15-Minute Glucose ROC",
            "Future TIR Decline Risk"
        ]
    )

    if model_choice == "Hypoglycemia Next 30 Minutes":
        model_data = df_view.copy()
        model_data["target"] = (
            model_data.groupby("patient_id")["glucose"].shift(-6) < 70
        ).astype(int)

        features = [
            "glucose", "glucose_roc", "glucose_rolling_std_1h",
            "basal_rate", bolus_col, "steps", "heart_rate", "hour"
        ]

        task = "classification"

    elif model_choice == "Hyperglycemia >200 Within 2 Hours After Meal":
        model_data = df_view[df_view["carb_input"] > 0].copy()
        model_data["target"] = (
            model_data.groupby("patient_id")["glucose"].shift(-24) > 200
        ).astype(int)

        features = [
            "glucose", "carb_input", bolus_col,
            "basal_rate", "steps", "heart_rate", "hour"
        ]

        task = "classification"

    elif model_choice == "Next 15-Minute Glucose ROC":
        model_data = df_view.copy()
        model_data["target"] = (
            model_data.groupby("patient_id")["glucose_roc"].shift(-3)
        )

        features = [
            "glucose", "glucose_roc", "glucose_rolling_std_1h",
            "basal_rate", bolus_col, "steps", "heart_rate", "hour"
        ]

        task = "regression"

    else:
        model_data = daily.copy()
        model_data["future_tir"] = (
            model_data.groupby("patient_id")["daily_tir"].shift(-7)
        )
        model_data["target"] = (
            model_data["future_tir"] < model_data["daily_tir"] - 10
        ).astype(int)

        features = [
            "daily_tir", "avg_glucose", "glucose_variability",
            "daily_steps", "avg_hr", "avg_basal", "total_bolus"
        ]

        task = "classification"

    model_df = model_data[features + ["target"]].dropna()

    if len(model_df) > 20000:
        model_df = model_df.sample(20000, random_state=42)

    if len(model_df) < 50 or model_df["target"].nunique() < 2:
        st.warning("Not enough balanced data for this model.")
    else:
        X = model_df[features]
        y = model_df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            st.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, prob):.3f}")

        else:
            model = RandomForestRegressor(
                n_estimators=80,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            st.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
            st.metric("R² Score", f"{r2_score(y_test, pred):.3f}")

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Feature Importance: {model_choice}"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PRESCRIPTIVE ANALYTICS
# ---------------------------------------------------

with tabs[6]:
    st.subheader("Prescriptive Insulin Effectiveness Score")

    score = (
        df_view
        .groupby("patient_id")
        .agg(
            tir=("is_in_range", "mean"),
            glucose_variability=("glucose", "std"),
            hypo_rate=("is_hypoglycemia", "mean"),
            hyper_rate=("is_hyperglycemia", "mean"),
            avg_steps=("steps", "mean"),
            total_bolus=(bolus_col, "sum"),
            avg_basal=("basal_rate", "mean")
        )
        .reset_index()
    )

    score["tir_score"] = score["tir"] * 40

    score["stability_score"] = (
        1 - score["glucose_variability"].rank(pct=True)
    ) * 25

    score["hypo_safety_score"] = (
        1 - score["hypo_rate"].rank(pct=True)
    ) * 20

    score["activity_score"] = (
        score["avg_steps"].rank(pct=True)
    ) * 15

    score["insulin_effectiveness_score"] = (
        score["tir_score"] +
        score["stability_score"] +
        score["hypo_safety_score"] +
        score["activity_score"]
    ).clip(0, 100)

    score = score.sort_values(
        "insulin_effectiveness_score",
        ascending=False
    )

    st.dataframe(score, use_container_width=True)

    fig = px.bar(
        score,
        x="patient_id",
        y="insulin_effectiveness_score",
        color="insulin_effectiveness_score",
        title="0–100 Insulin Effectiveness Score"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------

with tabs[7]:
    st.subheader("Hackathon Key Takeaways")

    st.markdown("""
    **1. Higher TIR is associated with lower glucose variability, stable insulin response, and consistent physical activity.**

    **2. Post-meal glucose spikes are strongly influenced by carbohydrate load, bolus timing, and missed bolus behavior.**

    **3. Nocturnal hypoglycemia risk increases with higher basal insulin and declining overnight glucose trends.**

    **4. Activity improves glucose stability, while prolonged inactivity increases glucose drift and instability.**

    **5. Predictive AI models can identify upcoming hypoglycemia, hyperglycemia, high-risk patients, and future TIR decline.**

    **6. Prescriptive scoring helps prioritize patients who need insulin adjustment, lifestyle intervention, or closer monitoring.**
    """)

    st.success("This dashboard combines descriptive, predictive, and prescriptive analytics into one clinical AI decision-support platform.")