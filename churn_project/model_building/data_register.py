import pandas as pd

RAW_PATH = "churn_project/data/Customer_Churn_Data.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "AccountID", "Churn", "Tenure", "City_Tier", "CC_Contacted_LY",
    "Payment", "Gender", "Service_Score", "Account_user_count",
    "account_segment", "CC_Agent_Score", "Marital_Status", "rev_per_month",
    "Complain_ly", "rev_growth_yoy", "coupon_used_for_payment",
    "Day_Since_CC_connect", "cashback", "Login_device",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Churn distribution:")
print(df["Churn"].value_counts())
