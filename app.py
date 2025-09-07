# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

import os
print(os.getcwd())  # shows current working dir
print(os.listdir()) # lists files in that folder

st.set_page_config(page_title="Purchase Prediction", page_icon="🛒", layout="centered")
st.title("🛒 Customer Purchase Prediction")

# -----------------------------
# Load model + encoders + scaler
# -----------------------------
model = tf.keras.models.load_model("purchasePredictionModel.h5")

with open("label_encoder_gender1.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_product1.pkl", "rb") as f:
    onehot_encoder_product = pickle.load(f)

with open("onehot_encoder_device1.pkl", "rb") as f:
    onehot_encoder_device = pickle.load(f)

with open("onehot_encoder_region1.pkl", "rb") as f:
    onehot_encoder_region = pickle.load(f)

with open("onehot_encoder_source1.pkl", "rb") as f:
    onehot_encoder_source = pickle.load(f)

with open("onehot_encoder_segment1.pkl", "rb") as f:
    onehot_encoder_segment = pickle.load(f)

with open("scaler1.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# UI helpers (choices from encoders to avoid unseen categories)
# -----------------------------
gender_choices = list(label_encoder_gender.classes_)                     # e.g., ['Female', 'Male']
product_choices = list(onehot_encoder_product.categories_[0])            # e.g., ['Electronics', 'Fashion', ...]
device_choices  = list(onehot_encoder_device.categories_[0])             # e.g., ['Mobile', 'Desktop', 'Tablet']
region_choices  = list(onehot_encoder_region.categories_[0])             # e.g., ['North', 'South', 'East', 'West']
source_choices  = list(onehot_encoder_source.categories_[0])             # e.g., ['Organic', 'Paid Ads', ...]
segment_choices = list(onehot_encoder_segment.categories_[0])            # e.g., ['Regular', 'Premium', 'VIP']

with st.form("purchase_form"):
    st.subheader("Input Features")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=90, value=35, step=1)
        gender = st.selectbox("Gender", gender_choices, index=gender_choices.index("Male") if "Male" in gender_choices else 0)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=75000.0, step=100.0)
        num_purchases = st.slider("Number of Purchases", min_value=0, max_value=50, value=12, step=1)
        time_spent = st.number_input("Time Spent on Website (minutes)", min_value=0.0, value=42.5, step=0.5)
        loyalty = st.selectbox("Loyalty Program (0/1)", [0, 1], index=1)
        discounts = st.slider("Discounts Availed", min_value=0, max_value=20, value=3, step=1)
        tenure_years = st.number_input("Customer Tenure (years)", min_value=0.0, value=2.4, step=0.1)

    with col2:
        product = st.selectbox("Product Category", product_choices, index=product_choices.index("Electronics") if "Electronics" in product_choices else 0)
        device = st.selectbox("Preferred Device", device_choices, index=device_choices.index("Mobile") if "Mobile" in device_choices else 0)
        region = st.selectbox("Region", region_choices, index=region_choices.index("South") if "South" in region_choices else 0)
        last_days = st.number_input("Last Purchase (days ago)", min_value=0, value=15, step=1)
        sessions = st.slider("Session Count", min_value=1, max_value=60, value=5, step=1)
        source = st.selectbox("Referral Source", source_choices, index=source_choices.index("Organic") if "Organic" in source_choices else 0)
        segment = st.selectbox("Customer Segment", segment_choices, index=segment_choices.index("Premium") if "Premium" in segment_choices else 0)
        satisfaction = st.slider("Customer Satisfaction (1-5)", min_value=1, max_value=5, value=4, step=1)

    threshold = st.slider("Decision Threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05, help="Probability above which we predict 'Likely to Purchase'.")

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Build input row in the SAME order as training
# -----------------------------
def build_feature_row():
    # Base numeric/label-encoded fields (same as training BEFORE OHE concat)
    base = pd.DataFrame([{
        "Age": age,
        "Gender": label_encoder_gender.transform([gender])[0],  # label-encoded
        "AnnualIncome": annual_income,
        "NumberOfPurchases": num_purchases,
        "TimeSpentOnWebsite": time_spent,
        "LoyaltyProgram": int(loyalty),
        "DiscountsAvailed": int(discounts),
        "CustomerTenureYears": float(tenure_years),
        "LastPurchaseDaysAgo": int(last_days),
        "SessionCount": int(sessions),
        "CustomerSatisfaction": int(satisfaction),
    }])

    # One-hot blocks in the SAME order you trained:
    # ProductCategory -> PreferredDevice -> Region -> ReferralSource -> CustomerSegment
    prod_ohe = onehot_encoder_product.transform([[product]]).toarray()
    dev_ohe  = onehot_encoder_device.transform([[device]]).toarray()
    reg_ohe  = onehot_encoder_region.transform([[region]]).toarray()
    src_ohe  = onehot_encoder_source.transform([[source]]).toarray()
    seg_ohe  = onehot_encoder_segment.transform([[segment]]).toarray()

    prod_df = pd.DataFrame(prod_ohe, columns=onehot_encoder_product.get_feature_names_out(["ProductCategory"]))
    dev_df  = pd.DataFrame(dev_ohe,  columns=onehot_encoder_device.get_feature_names_out(["PreferredDevice"]))
    reg_df  = pd.DataFrame(reg_ohe,  columns=onehot_encoder_region.get_feature_names_out(["Region"]))
    src_df  = pd.DataFrame(src_ohe,  columns=onehot_encoder_source.get_feature_names_out(["ReferralSource"]))
    seg_df  = pd.DataFrame(seg_ohe,  columns=onehot_encoder_segment.get_feature_names_out(["CustomerSegment"]))

    # Concatenate base + one-hot blocks
    row = pd.concat([base.reset_index(drop=True), prod_df, dev_df, reg_df, src_df, seg_df], axis=1)
    return row

# -----------------------------
# Predict
# -----------------------------
if submitted:
    try:
        row = build_feature_row()
        # Scale exactly like training
        row_scaled = scaler.transform(row)
        proba = float(model.predict(row_scaled, verbose=0)[0][0])

        st.markdown(f"**Purchase Probability:** `{proba:.4f}`")
        label = "✅ Likely to Purchase" if proba >= threshold else "❌ Not Likely to Purchase"
        st.subheader(label)

        with st.expander("See encoded feature vector (debug)"):
            st.write("Shape:", row.shape)
            st.dataframe(row)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Tip: Make sure the encoders/scaler and the model were trained on the SAME schema and category sets.")
