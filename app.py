# ======================================================================
# 🏨 app.py — Hotel Booking Cancellation Prediction Dashboard
# ======================================================================

import streamlit as st
import pandas as pd
import joblib
import os
import json
import plotly.express as px
import plotly.io as pio

# ======================================================================
# 📁 BASE DIRECTORY (IMPORTANT)
# ======================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# 📊 LOAD MODEL METRICS (SAFE)
# ======================================================================

@st.cache_data
def load_metrics():
    metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)

metrics = load_metrics()

# ======================================================================
# 🤖 LOAD MODEL (CACHED — VERY IMPORTANT)
# ======================================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "hotel_cancellation_xgb_pipeline.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

# ======================================================================
# 📊 LOAD DATASET (CACHED — REQUIRED)
# ======================================================================

@st.cache_data
def load_dataset():
    data_path = os.path.join(BASE_DIR, "hotel_bookings.csv")
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


# ======================================================================
# 🌈 PAGE CONFIGURATION
# ======================================================================

st.set_page_config(
    page_title="🏨 Hotel Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)


# ======================================================================
# 🎫 BOOKING TICKET STYLES (GLOBAL — SAFE PLACE)
# ======================================================================

st.markdown("""
<style>
.ticket {
    border-radius: 16px;
    padding: 24px;
    margin-top: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    font-family: "Segoe UI", sans-serif;
    border-left: 8px solid;
}

.ticket-success {
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
    border-color: #2e7d32;
}

.ticket-cancel {
    background: linear-gradient(135deg, #fdecea, #ffffff);
    border-color: #c62828;
}

.ticket-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
}

.ticket-success .ticket-title {
    color: #2e7d32;
}

.ticket-cancel .ticket-title {
    color: #c62828;
}

.ticket-body {
    font-size: 16px;
    color: #333;
}

.ticket-prob {
    font-size: 18px;
    color: #0d47a1;
    font-weight: 700;
    margin-top: 12px;
}

.ticket-footer {
    margin-top: 14px;
    font-size: 13px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# 🎨 PLOTLY THEME
# ======================================================================

pio.templates.default = "plotly_white"
pio.templates["plotly_white"].layout.update({
    "paper_bgcolor": "rgba(240,250,252,1)",
    "plot_bgcolor": "rgba(240,250,252,1)",
    "font": {"color": "#083b46", "family": "Poppins"},
    "xaxis": {"gridcolor": "#d4f1f4"},
    "yaxis": {"gridcolor": "#d4f1f4"},
    "title": {"font": {"size": 20, "color": "#00796b"}}
})

# ======================================================================
# 🎨 GLOBAL CSS
# ======================================================================

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0097a7, #006064);
    border-right: 2px solid #00acc1;
}
.metric-card {
    background: rgba(255,255,255,0.9);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# 🧭 SIDEBAR
# ======================================================================

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Overview", "📊 Dataset Overview", "🔍 EDA", "📈 Model Metrics", "❌ Prediction"]
)

# ======================================================================
# 🏠 OVERVIEW
# ======================================================================

if page == "🏠 Overview":
    st.title("🏨 Hotel Booking Cancellation Prediction Dashboard")

    st.markdown("""
    ### 📘 Project Overview
    This application predicts whether a **hotel booking is likely to be cancelled**
    using a **Machine Learning classification model** trained on real hotel booking data.

    The model helps hotel management and travel platforms to:
    - 📉 Reduce revenue loss from cancellations  
    - 🧠 Understand booking behavior patterns  
    - 🎯 Take preventive actions in advance  

    ---
    ### ⚙️ How the System Works
    1. 📊 Historical hotel booking data is preprocessed  
    2. 🧹 Categorical & numerical features are cleaned and encoded  
    3. 🤖 A trained **XGBoost model** learns cancellation patterns  
    4. 🔮 New booking details are used to predict cancellation probability  

    ---
    ### 🧾 Key Features Used for Prediction
    - 🏨 Hotel type (City / Resort)  
    - 📅 Arrival month  
    - 🍽 Meal plan  
    - 💳 Deposit type  
    - ⏳ Lead time before arrival  
    - 👨‍👩‍👧 Number of adults & children  
    - 💰 Average Daily Rate (ADR)  
    - 🔁 Previous cancellations  
    - ⭐ Special service requests  

    ---
    ### 🧪 Model Information
    - **Algorithm:** XGBoost Classifier  
    - **Pipeline:** Preprocessing + Model (scikit-learn)  
    - **Output:**  
      - `0` → Booking NOT likely to be cancelled  
      - `1` → Booking likely to be cancelled  

    ---
    👉 Use the **❌ Prediction** tab from the sidebar to test live predictions.
    """)

    st.divider()

    data_path = os.path.join(BASE_DIR, "hotel_bookings.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📄 Total Records", f"{len(df):,}")
        c2.metric("❌ Cancelled", int(df["is_canceled"].sum()))
        c3.metric("✅ Not Cancelled", int((df["is_canceled"] == 0).sum()))
        c4.metric(
            "📉 Cancellation Rate",
            f"{df['is_canceled'].mean() * 100:.2f}%"
        )

    st.info("ℹ️ This dashboard is built using **Python, Streamlit, Scikit-Learn, and XGBoost**.")
# ======================================================================
# 📊 DATASET OVERVIEW
# ======================================================================

elif page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")

    df = load_dataset()
    if df is None:
        st.error("❌ Dataset file not found!")
    else:
        # --------------------------------------------------
        # Dataset Shape
        # --------------------------------------------------
        st.subheader("📐 Dataset Shape")
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        st.divider()

        # --------------------------------------------------
        # First 5 Rows
        # --------------------------------------------------
        st.subheader("🔍 First 5 Rows (Head)")
        st.dataframe(df.head(5), use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # Dataset Info (info() equivalent)
        # --------------------------------------------------
        st.subheader("ℹ️ Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum(),
            "Data Type": df.dtypes.astype(str)
        })

        st.dataframe(info_df, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # Summary Statistics
        # --------------------------------------------------
        st.subheader("📘 Summary Statistics (describe)")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # Missing Values Summary
        # --------------------------------------------------
        st.subheader("❗ Missing Values Summary")

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values("Missing Values", ascending=False)

        st.dataframe(missing_df, use_container_width=True)


# ======================================================================
# 🔍 EDA
# ======================================================================

elif page == "🔍 EDA":
    st.header("🔍 Exploratory Data Analysis")

    df = load_dataset()
    if df is None:
        st.error("❌ Dataset file not found!")
    else:
        # --------------------------------------------------
        # 1. Cancellation Distribution
        # --------------------------------------------------
        st.subheader("📌 Booking Cancellation Distribution")

        fig = px.histogram(
            df,
            x="is_canceled",
            color="is_canceled",
            labels={"is_canceled": "Cancelled (0 = No, 1 = Yes)"},
            color_discrete_sequence=["#00796b", "#d32f2f"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # 2. Cancellation by Hotel Type
        # --------------------------------------------------
        st.subheader("🏨 Cancellation by Hotel Type")

        fig = px.histogram(
            df,
            x="hotel",
            color="is_canceled",
            barmode="group",
            color_discrete_sequence=["#00796b", "#d32f2f"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # 3. Lead Time vs Cancellation
        # --------------------------------------------------
        st.subheader("⏳ Lead Time vs Cancellation")

        fig = px.box(
            df,
            x="is_canceled",
            y="lead_time",
            color="is_canceled",
            labels={"is_canceled": "Cancelled"},
            color_discrete_sequence=["#00796b", "#d32f2f"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # 4. ADR vs Cancellation
        # --------------------------------------------------
        st.subheader("💰 Average Daily Rate (ADR) vs Cancellation")

        fig = px.box(
            df,
            x="is_canceled",
            y="adr",
            color="is_canceled",
            labels={"is_canceled": "Cancelled"},
            color_discrete_sequence=["#00796b", "#d32f2f"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # 5. Monthly Cancellation Trend
        # --------------------------------------------------
        st.subheader("📅 Monthly Cancellation Trend")

        month_order = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ]

        fig = px.histogram(
            df,
            x="arrival_date_month",
            color="is_canceled",
            category_orders={"arrival_date_month": month_order},
            color_discrete_sequence=["#00796b", "#d32f2f"]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "🔎 **Insight:** Bookings with higher lead time and higher ADR "
            "show a greater tendency to cancel."
        )
# ======================================================================
# 📈 MODEL METRICS
# ======================================================================

elif page == "📈 Model Metrics":
    st.header("📈 Model Performance Metrics")

    if metrics is None:
        st.error("❌ Model metrics file not found. Please ensure `model_metrics.json` exists.")
    else:
        st.markdown("""
        This section displays the **actual performance metrics**
        obtained during offline model training and testing.

        ✔ Metrics are computed once during training  
        ✔ No retraining happens in Streamlit  
        """)

        st.divider()

        # --------------------------------------------------
        # Best Model
        # --------------------------------------------------
        st.subheader("🏆 Best Model Selected")
        st.success(f"**{metrics['best_model']}**")

        st.divider()

        # --------------------------------------------------
        # Training vs Testing Scores
        # --------------------------------------------------
        st.subheader("📊 Training vs Testing Performance")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train Accuracy", f"{metrics['train_accuracy']*100:.2f}%")
        c2.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
        c3.metric("Train ROC-AUC", f"{metrics['train_roc_auc']:.2f}")
        c4.metric("Test ROC-AUC", f"{metrics['test_roc_auc']:.2f}")

        st.divider()

        # --------------------------------------------------
        # Classification Metrics (Test Data)
        # --------------------------------------------------
        st.subheader("📌 Classification Metrics (Test Set)")

        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "ROC-AUC"],
            "Score": [
                f"{metrics['test_accuracy']*100:.2f}%",
                f"{metrics['precision']*100:.2f}%",
                f"{metrics['recall']*100:.2f}%",
                f"{metrics['test_roc_auc']:.2f}"
            ]
        })

        st.dataframe(metric_df, use_container_width=True)

        st.divider()

        # --------------------------------------------------
        # Interpretation
        # --------------------------------------------------
        st.subheader("🧠 Interpretation")

        st.info("""
        🔹 The small gap between training and testing scores indicates
        **good generalization**.

        🔹 ROC–AUC above **0.80** shows strong discrimination ability.

        🔹 XGBoost outperformed other models (Logistic Regression,
        Random Forest) during experimentation.
        """)


# ======================================================================
# ❌ MANUAL PREDICTION
# ======================================================================

elif page == "❌ Prediction":
    st.header("❌ Manual Booking Cancellation Prediction")

    if model is None:
        st.error("❌ Model not loaded. Please check model file.")
    else:
        st.markdown("""
        Enter booking details manually to predict whether the reservation  
        will be **CONFIRMED or CANCELLED**.
        """)

        col1, col2 = st.columns(2)

        with col1:
            hotel = st.selectbox("🏨 Hotel Type", ["City Hotel", "Resort Hotel"])
            arrival_month = st.selectbox(
                "📅 Arrival Month",
                ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"]
            )
            meal = st.selectbox("🍽 Meal Plan", ["BB", "HB", "FB", "SC"])
            deposit_type = st.selectbox(
                "💳 Deposit Type",
                ["No Deposit", "Refundable", "Non Refund"]
            )

        with col2:
            lead_time = st.number_input("⏳ Lead Time (days)", 0, 365, 50)
            adults = st.number_input("👨 Adults", 1, 5, 2)
            children = st.number_input("👧 Children", 0, 5, 0)
            adr = st.number_input("💰 ADR", 0.0, 500.0, 100.0)
            previous_cancellations = st.number_input("🔁 Previous Cancellations", 0, 10, 0)
            total_of_special_requests = st.number_input("⭐ Special Requests", 0, 5, 0)

        st.divider()

if st.button("🔮 Predict Cancellation"):
    with st.spinner("Analyzing booking details..."):
        input_df = pd.DataFrame([{
            "hotel": hotel,
            "arrival_date_month": arrival_month,
            "meal": meal,
            "deposit_type": deposit_type,
            "lead_time": lead_time,
            "adults": adults,
            "children": children,
            "adr": adr,
            "previous_cancellations": previous_cancellations,
            "total_of_special_requests": total_of_special_requests
        }])

        prob = model.predict_proba(input_df)[0][1]

    THRESHOLD = 0.85  # <-- tuning point

    if prob >= THRESHOLD:
        st.markdown(f"""
        <div class="ticket ticket-cancel">
            <div class="ticket-title">❌ BOOKING CANCELLED</div>
            <div class="ticket-body">
                This booking shows a <b>high risk of cancellation</b>.
            </div>
            <div class="ticket-prob">
                Cancellation Probability: <b>{prob:.2%}</b>
            </div>
            <div class="ticket-footer">
                Status generated by ML prediction system
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="ticket ticket-success">
            <div class="ticket-title">✅ BOOKING CONFIRMED</div>
            <div class="ticket-body">
                This booking is <b>very likely to be honored</b>.
            </div>
            <div class="ticket-prob">
                Confidence Score: <b>{(1 - prob):.2%}</b>
            </div>
            <div class="ticket-footer">
                Status generated by ML prediction system
            </div>
        </div>
        """, unsafe_allow_html=True)
