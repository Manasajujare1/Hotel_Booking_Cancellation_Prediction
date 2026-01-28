# 🏨 Hotel Booking Cancellation Prediction

An end-to-end **Machine Learning project** that predicts whether a hotel booking is likely to be **cancelled or not**, built using **XGBoost**, deployed with **Streamlit**, and designed for real-world usage.

---

## 📌 Project Overview

Hotel booking cancellations cause significant revenue loss for hotels.  
This project aims to **predict booking cancellations in advance** so hotels can take proactive actions such as overbooking strategies, dynamic pricing, or targeted customer engagement.

The solution includes:
- Data analysis & preprocessing
- Machine learning pipeline
- Model evaluation
- Interactive Streamlit dashboard
- Cloud deployment

---

## 🎯 Problem Statement

> Predict whether a hotel booking will be **cancelled (1)** or **not cancelled (0)** using historical booking data.

This is a **binary classification problem**.

---

## 📂 Dataset

- **Source:** Hotel booking dataset (CSV)
- **Rows:** ~119,000
- **Target Variable:** `is_canceled`

### Key Features Used:
- `hotel`
- `lead_time`
- `arrival_date_month`
- `adults`
- `children`
- `meal`
- `market_segment`
- `adr`

---

## 🧠 Machine Learning Approach

### 🔹 Model Used
- **XGBoost Classifier**

### 🔹 Why XGBoost?
- Handles non-linear relationships well
- Regularization prevents overfitting
- High performance on structured/tabular data
- Industry-grade algorithm

### 🔹 Pipeline Components
- Categorical encoding (OneHotEncoder)
- Feature preprocessing using `ColumnTransformer`
- XGBoost model integrated into a single pipeline

---

## 📊 Model Evaluation

- **Primary Metric:** ROC–AUC
- Other metrics evaluated:
  - Accuracy
  - Precision
  - Recall

The final model was selected based on **best ROC–AUC score**.

---

## 🖥️ Streamlit Dashboard Features

The deployed dashboard provides:

### 🏠 Overview
- Total bookings
- Cancelled bookings
- Cancellation rate

### 📊 Dataset Overview
- Dataset shape
- Missing values
- Sample records
- Summary statistics

### 🔍 Exploratory Data Analysis (EDA)
- Cancellation distribution
- Lead time vs cancellation
- Interactive Plotly visualizations

### ❌ Live Prediction
- User input form
- Real-time prediction
- Cancellation probability

---

## 🚀 Deployment

- **Platform:** Streamlit Community Cloud
- **Deployment Type:** GitHub-based
- **Model Loading:** `joblib`

---

## 🧾 Project Structure

-- app.py
-- requirements.txt
-- hotel_bookings.csv
-- hotel_cancellation_xgb_pipeline.pkl
-- README.md
