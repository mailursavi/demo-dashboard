# app.py
# Streamlit Diabetes Analytics Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

st.set_page_config(
    page_title="Diabetes AI Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.5rem;}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes AI Analytics Dashboard")
st.caption("Interactive CGM, insulin, meal, activity, sleep, predictive and prescriptive analytics")

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_hupa_diabetes_recent (1).xlsb")
    demo = pd.read_csv("cleaned_demographics(1).csv")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    if "patient_id" in demo.columns:
        df = df.merge(demo, on="patient_id", how="left")

    return df

df = load_data()
        

# =========================
# PREPROCESSING
# =========================

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time", "patient_id", "glucose"])
df = df.sort_values(["patient_id", "time"])

if demo is not None:
    df = df.merge(demo, on="patient_id", how="left")

bolus_col = "bolus_volume_delivered"
if bolus_col not in df.columns:
    bolus_col = "bolus"

df["date"] = df["time"].dt.date
df["hour"] = df["time"].dt.hour
df["is_weekend"] = df["time"].dt.dayofweek.isin([5, 6]).astype(int)
df["is_night"] = df["hour"].between(0, 5).astype(int)

if "glucose_roc" not in df.columns:
    df["glucose_roc"] = df.groupby("patient_id")["glucose"].diff()

if "glucose_rolling_std_1h" not in df.columns:
    df["glucose_rolling_std_1h"] = (
        df.groupby("patient_id")["glucose"]
        .rolling(12)
        .std()
        .reset_index(level=0, drop=True)
    )

if "glucose_rolling_mean_1h" not in df.columns:
    df["glucose_rolling_mean_1h"] = (
        df.groupby("patient_id")["glucose"]
        .rolling(12)
        .mean()
        .reset_index(level=0, drop=True)
    )

df["tir_flag"] = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).astype(int)
df["hypo_flag"] = (df["glucose"] < 70).astype(int)
df["hyper_flag"] = (df["glucose"] > 180).astype(int)

# =========================
# SIDEBAR FILTERS
# =========================

patients = sorted(df["patient_id"].dropna().unique())
selected_patients = st.sidebar.multiselect(
    "Select Patients",
    patients,
    default=patients[:5]
)

df_view = df[df["patient_id"].isin(selected_patients)].copy()

if df_view.empty:
    st.warning("Please select at least one patient.")
    st.stop()

# =========================
# KPI METRICS
# =========================

tir = df_view["tir_flag"].mean() * 100
hypo = df_view["hypo_flag"].mean() * 100
hyper = df_view["hyper_flag"].mean() * 100
avg_glucose = df_view["glucose"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Time-In-Range", f"{tir:.1f}%")
c2.metric("Hypoglycemia", f"{hypo:.1f}%")
c3.metric("Hyperglycemia", f"{hyper:.1f}%")
c4.metric("Average Glucose", f"{avg_glucose:.1f} mg/dL")

# =========================
# DAILY SUMMARY
# =========================

daily = (
    df_view
    .groupby(["patient_id", "date"])
    .agg(
        daily_tir=("tir_flag", "mean"),
        avg_glucose=("glucose", "mean"),
        glucose_variability=("glucose", "std"),
        daily_steps=("steps", "sum"),
        avg_hr=("heart_rate", "mean"),
        avg_basal=("basal_rate", "mean"),
        total_bolus=(bolus_col, "sum"),
        total_carbs=("carb_input", "sum"),
        hypo_rate=("hypo_flag", "mean"),
        hyper_rate=("hyper_flag", "mean")
    )
    .reset_index()
)

daily["daily_tir"] *= 100

# =========================
# TABS
# =========================

tabs = st.tabs([
    "📊 Overview",
    "🍽️ Meal & Bolus",
    "🏃 Activity",
    "🌙 Night Risk",
    "🧠 Predictive AI",
    "💊 Prescriptive Score",
    "📌 Insights"
])

# =========================
# TAB 1: OVERVIEW
# =========================

with tabs[0]:
    st.subheader("24-Hour Glucose Trend")

    fig = px.line(
        df_view,
        x="time",
        y="glucose",
        color="patient_id",
        title="Glucose Trend Over Time"
    )

    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=180, line_dash="dash", line_color="red")

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.box(
            df_view,
            x="patient_id",
            y="glucose",
            title="Glucose Distribution by Patient"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            daily,
            x="date",
            y="daily_tir",
            color="patient_id",
            title="Daily Time-In-Range"
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2: MEAL & BOLUS
# =========================

with tabs[1]:
    st.subheader("Meal, Carbohydrate, and Bolus Analysis")

    df_meal = df_view.copy()
    df_meal["glucose_next_2h"] = (
        df_meal.groupby("patient_id")["glucose"].shift(-24)
    )
    df_meal["post_meal_spike"] = (
        df_meal["glucose_next_2h"] - df_meal["glucose"]
    )

    meal_df = df_meal[df_meal["carb_input"] > 0].dropna(
        subset=["post_meal_spike", "carb_input", bolus_col]
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
            title="Carbs vs Post-Meal Spike"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.density_heatmap(
            meal_df,
            x="carb_input",
            y="post_meal_spike",
            nbinsx=30,
            nbinsy=30,
            title="Carb Intake vs Spike Density"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missed Bolus Detection")

    meal_df["missed_bolus"] = (
        (meal_df["carb_input"] > 20) &
        (meal_df[bolus_col] == 0) &
        (meal_df["glucose_next_2h"] > 180)
    ).astype(int)

    missed_summary = (
        meal_df["missed_bolus"]
        .value_counts()
        .reset_index()
    )
    missed_summary.columns = ["missed_bolus", "count"]

    fig = px.bar(
        missed_summary,
        x="missed_bolus",
        y="count",
        title="Missed Bolus Events"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: ACTIVITY
# =========================

with tabs[2]:
    st.subheader("Activity Impact on Glucose Stability")

    df_act = df_view.copy()

    df_act["glucose_drift_1h"] = (
        df_act.groupby("patient_id")["glucose"].diff(12)
    )

    df_act["activity_group"] = pd.cut(
        df_act["steps"],
        bins=[-1, 0, 50, 500, 100000],
        labels=["Sedentary", "Low", "Moderate", "High"]
    )

    activity_summary = (
        df_act
        .groupby("activity_group", observed=True)
        .agg(
            avg_glucose_drift=("glucose_drift_1h", "mean"),
            avg_instability=("glucose_rolling_std_1h", "mean"),
            avg_glucose=("glucose", "mean")
        )
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            activity_summary,
            x="activity_group",
            y="avg_instability",
            color="activity_group",
            title="Activity Level vs Glucose Instability"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df_act.sample(min(5000, len(df_act)), random_state=42),
            x="steps",
            y="glucose_drift_1h",
            color="activity_group",
            title="Steps vs 1-Hour Glucose Drift"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Steps vs TIR")

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

# =========================
# TAB 4: NIGHT RISK
# =========================

with tabs[3]:
    st.subheader("Nocturnal Hypoglycemia and Basal Rate Risk")

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
            title="Basal Rate Distribution by Nocturnal Hypoglycemia"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Early Morning Glucose Rise")

    dawn = df_view[df_view["hour"].between(0, 8)].copy()
    dawn_hourly = (
        dawn
        .groupby("hour")["glucose"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        dawn_hourly,
        x="hour",
        y="glucose",
        markers=True,
        title="Average Overnight to Morning Glucose Pattern"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 5: PREDICTIVE AI
# =========================

with tabs[4]:
    st.subheader("Predictive Models")

    model_choice = st.selectbox(
        "Choose Predictive Model",
        [
            "Hypoglycemia in Next 30 Minutes",
            "Glucose >200 Within 2 Hours After Meal",
            "Next 15-Minute Glucose ROC",
            "Daily TIR Decline Risk"
        ]
    )

    if model_choice == "Hypoglycemia in Next 30 Minutes":
        model_data = df_view.copy()
        model_data["glucose_next_30min"] = (
            model_data.groupby("patient_id")["glucose"].shift(-6)
        )
        model_data["target"] = (
            model_data["glucose_next_30min"] < 70
        ).astype(int)

        features = [
            "glucose", "glucose_roc", "glucose_rolling_std_1h",
            "basal_rate", bolus_col, "steps", "heart_rate", "hour"
        ]

        task_type = "classification"

    elif model_choice == "Glucose >200 Within 2 Hours After Meal":
        model_data = df_view[df_view["carb_input"] > 0].copy()
        model_data["glucose_next_2h"] = (
            model_data.groupby("patient_id")["glucose"].shift(-24)
        )
        model_data["target"] = (
            model_data["glucose_next_2h"] > 200
        ).astype(int)

        features = [
            "glucose", "carb_input", bolus_col,
            "basal_rate", "steps", "heart_rate", "hour"
        ]

        task_type = "classification"

    elif model_choice == "Next 15-Minute Glucose ROC":
        model_data = df_view.copy()
        model_data["target"] = (
            model_data.groupby("patient_id")["glucose_roc"].shift(-3)
        )

        features = [
            "glucose", "glucose_roc", "glucose_rolling_std_1h",
            "basal_rate", bolus_col, "steps", "heart_rate", "hour"
        ]

        task_type = "regression"

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

        task_type = "classification"

    model_df = model_data[features + ["target"]].dropna()

    if len(model_df) < 50 or model_df["target"].nunique() < 2:
        st.warning("Not enough balanced data for this model.")
    else:
        if len(model_df) > 20000:
            model_df = model_df.sample(20000, random_state=42)

        X = model_df[features]
        y = model_df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
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
            st.metric("R²", f"{r2_score(y_test, pred):.3f}")

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

# =========================
# TAB 6: PRESCRIPTIVE SCORE
# =========================

with tabs[5]:
    st.subheader("Insulin Effectiveness Composite Score")

    patient_score = (
        df_view
        .groupby("patient_id")
        .agg(
            tir=("tir_flag", "mean"),
            avg_glucose=("glucose", "mean"),
            glucose_variability=("glucose", "std"),
            hypo_rate=("hypo_flag", "mean"),
            hyper_rate=("hyper_flag", "mean"),
            avg_steps=("steps", "mean"),
            avg_basal=("basal_rate", "mean"),
            total_bolus=(bolus_col, "sum")
        )
        .reset_index()
    )

    patient_score["tir_score"] = patient_score["tir"] * 40
    patient_score["stability_score"] = (
        1 - patient_score["glucose_variability"].rank(pct=True)
    ) * 25
    patient_score["hypo_safety_score"] = (
        1 - patient_score["hypo_rate"].rank(pct=True)
    ) * 20
    patient_score["activity_score"] = (
        patient_score["avg_steps"].rank(pct=True)
    ) * 15

    patient_score["insulin_effectiveness_score"] = (
        patient_score["tir_score"] +
        patient_score["stability_score"] +
        patient_score["hypo_safety_score"] +
        patient_score["activity_score"]
    ).clip(0, 100)

    patient_score = patient_score.sort_values(
        "insulin_effectiveness_score",
        ascending=False
    )

    st.dataframe(patient_score, use_container_width=True)

    fig = px.bar(
        patient_score,
        x="patient_id",
        y="insulin_effectiveness_score",
        color="insulin_effectiveness_score",
        title="Patient Insulin Effectiveness Score"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        patient_score,
        x="glucose_variability",
        y="tir",
        size="avg_steps",
        color="insulin_effectiveness_score",
        hover_data=["patient_id"],
        title="TIR vs Glucose Variability with Activity"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 7: INSIGHTS
# =========================

with tabs[6]:
    st.subheader("Executive Insights")

    st.markdown("""
    **1. Glucose Stability:** Patients with higher Time-In-Range usually show lower glucose variability and fewer extreme highs/lows.  
    **2. Meal Impact:** Higher carbohydrate loads are associated with larger post-meal glucose spikes, especially when bolus timing or dose is insufficient.  
    **3. Activity Impact:** Higher daily steps generally improve TIR and reduce instability, while prolonged inactivity increases glucose drift risk.  
    **4. Night Risk:** Higher basal rates and declining nighttime glucose trends may increase nocturnal hypoglycemia risk.  
    **5. Predictive AI:** CGM trends, rolling variability, heart rate, activity, insulin, and carbs can support early risk prediction and personalized intervention.
    """)

    st.success("Dashboard ready for hackathon demo: descriptive + predictive + prescriptive diabetes analytics.")
