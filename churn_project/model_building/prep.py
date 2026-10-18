import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("churn_project/data/Customer_Churn_Data.csv")

# ---------------------------------------------------------------
# 1. Clean special characters (#, $, @, +, *) that appear in several
#    columns and turn them into proper NaN values.
# ---------------------------------------------------------------
special_chars = ["#", "$", "@", "+", "*"]
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].replace(special_chars, np.nan, regex=False)

# ---------------------------------------------------------------
# 2. Standardise gender labels
# ---------------------------------------------------------------
df["Gender"] = df["Gender"].replace({"F": "Female", "M": "Male"})

# ---------------------------------------------------------------
# 3. Replace invalid login device tags
# ---------------------------------------------------------------
df["Login_device"] = df["Login_device"].replace({"&&&&": np.nan})

# ---------------------------------------------------------------
# 4. Coerce numeric columns (which now contain NaNs) to numeric dtype
# ---------------------------------------------------------------
numeric_cols = [
    "Tenure", "City_Tier", "CC_Contacted_LY", "Service_Score",
    "Account_user_count", "CC_Agent_Score", "rev_per_month",
    "rev_growth_yoy", "coupon_used_for_payment",
    "Day_Since_CC_connect", "cashback",
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------------
# 5. Drop the unique identifier — it is not a predictive feature
# ---------------------------------------------------------------
df.drop(columns=["AccountID"], inplace=True)

# ---------------------------------------------------------------
# 6. Simple imputation so the pipeline can run end-to-end
#    (numeric → median, categorical → mode)
# ---------------------------------------------------------------
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values, so training and serving stay consistent.

target = "Churn"
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) churn ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Churn distribution in train:")
print(ytrain.value_counts())
