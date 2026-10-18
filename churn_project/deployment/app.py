import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_churn_model_v1.joblib")
model = joblib.load(model_path)

st.title("ChurnGuard — Customer Churn Prediction")
st.write("Fill the account details below to predict if the customer will churn")

# Collect user input
Tenure = st.slider("Tenure (months)", 0, 60, 6)
City_Tier = st.selectbox("City Tier", [1, 2, 3])
CC_Contacted_LY = st.slider("Customer Care Contacts (last 12m)", 0, 50, 10)
Payment = st.selectbox(
    "Preferred Payment Mode",
    ["Debit Card", "Credit Card", "UPI", "E wallet", "Cash on Delivery"]
)
Gender = st.selectbox("Gender", ["Male", "Female"])
Service_Score = st.slider("Service Satisfaction Score", 1, 5, 3)
Account_user_count = st.slider("Number of Users on Account", 1, 10, 3)
account_segment = st.selectbox(
    "Account Segment",
    ["Regular", "Regular Plus", "Super", "Super Plus", "HNI"]
)
CC_Agent_Score = st.slider("CC Agent Satisfaction Score", 1, 5, 3)
Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
rev_per_month = st.number_input("Monthly Revenue", min_value=0.0, value=5000.0)
Complain_ly = st.selectbox("Complaint Raised (last 12m)?", [0, 1])
rev_growth_yoy = st.slider("Revenue Growth YoY (%)", 0, 50, 15)
coupon_used_for_payment = st.slider("Coupons Used (last 12m)", 0, 20, 1)
Day_Since_CC_connect = st.slider("Days Since Last CC Contact", 0, 30, 5)
cashback = st.number_input("Monthly Cashback", min_value=0.0, value=150.0)
Login_device = st.selectbox("Preferred Login Device", ["Mobile", "Computer"])

# Prepare input data
input_data = pd.DataFrame([{
    "Tenure": Tenure,
    "City_Tier": City_Tier,
    "CC_Contacted_LY": CC_Contacted_LY,
    "Payment": Payment,
    "Gender": Gender,
    "Service_Score": Service_Score,
    "Account_user_count": Account_user_count,
    "account_segment": account_segment,
    "CC_Agent_Score": CC_Agent_Score,
    "Marital_Status": Marital_Status,
    "rev_per_month": rev_per_month,
    "Complain_ly": Complain_ly,
    "rev_growth_yoy": rev_growth_yoy,
    "coupon_used_for_payment": coupon_used_for_payment,
    "Day_Since_CC_connect": Day_Since_CC_connect,
    "cashback": cashback,
    "Login_device": Login_device,
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    result = "is likely to CHURN" if pred == 1 else "is likely to STAY"
    st.write(f"Prediction: Customer {result}")
    st.write(f"Churn probability: {prob:.2%}")
