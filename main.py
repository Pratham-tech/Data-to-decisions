import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Data
df = pd.read_csv("Telco_customer_churn.csv")

# --- PARAMETERS ---
EXPECTED_MONTHS = 6
RETENTION_SUCCESS_RATE = 0.4
OFFER_COST = 60

# --- DATA PREPROCESSING ---

# 1. Drop constant/redundant columns
drop_cols = [
    "Country", "State", "Count",       # Constants
    "City", "Zip Code", "Lat Long",    # Location details (too cardincal/redundant)
    "Latitude", "Longitude",           # Location details
    "Churn Label", "Churn Score",      # Leaky/Target related
    "CLTV", "Churn Reason",            # Leaky/Post-event info
    "CustomerID"                       # ID
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 2. Fix Total Charges
df["Total Charges"] = df["Total Charges"].str.strip()
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

# 3. Target Variable
y = df["Churn Value"]
X = df.drop(columns=["Churn Value"])

# 4. Feature Engineering: Estimate Customer Value for Profit Calc
# We'll use the original dataframe's Monthly Charges for this calculation later
customer_value_all = df["Monthly Charges"] * EXPECTED_MONTHS

# 5. Encoding
# Identify categorical columns
cat_cols = X.select_dtypes(include=['object']).columns

# One-Hot Encoding for categorical variables
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# --- MODEL TRAINING ---

# Split Data
X_train, X_test, y_train, y_test, value_train, value_test = train_test_split(
    X_encoded, y, customer_value_all, test_size=0.2, random_state=42
)

# Train XGBoost
model = xgb.XGBClassifier(
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# Predict Probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# --- PROFIT ANALYSIS ---

def calculate_profit(y_true, y_prob, customer_value, threshold):
    """
    Calculates total profit based on retention offer strategy.
    
    Logic:
    - TP (Churn=1, Pred=1): We offer retention. Success rate * Value - Cost.
    - FN (Churn=1, Pred=0): We do nothing. They leave. Loss = -Value.
    - FP (Churn=0, Pred=1): We offer retention. They stay anyway. Loss = -Cost.
    - TN (Churn=0, Pred=0): We do nothing. They stay. No extra profit/loss change (baseline).
    """
    profit = 0
    # Reset indices to align interaction
    y_true = y_true.values
    customer_value = customer_value.values
    
    for i in range(len(y_true)):
        actual = y_true[i]
        prob = y_prob[i]
        val = customer_value[i]
        
        offer = prob >= threshold
        
        if actual == 1:
            if offer:
                # We try to retain. 
                # Profit = (Chance they stay * LTV) - Cost of offer
                # If they leave despite offer, we lose val (captured in 'Chance they stay' expectation?)
                # Simplified: 
                #  Success: Saved 'val' (which was about to be lost) -> Gain = val
                #  Fail: 'val' is lost.
                #  Cost: always paid.
                # Expected Gain = (SuccessRate * val) - Cost.
                # Note: If we didn't offer, we lose 'val' for sure (-val).
                # So the *impact* of the action vs baseline of doing nothing (-val) is:
                #  Action Value = (SuccRate * val) - Cost
                #  No Action Value = 0 (relative to losing them? No, let's stick to absolute/impact terms)
                
                # Let's align with the user's previous logic:
                # if actual=1 and offer: profit += (RET_RATE * val) - COST
                profit += (RETENTION_SUCCESS_RATE * val) - OFFER_COST
            else:
                # actual=1 and no offer: we lose the customer value
                profit -= val
        else: # actual == 0 (Loyal customer)
            if offer:
                # We wasted money on a loyal customer
                profit -= OFFER_COST
            # else: no offer, loyal customer. No change to P&L deviation.
            
    return profit

thresholds = np.linspace(0, 1, 101)
profits = []

for t in thresholds:
    p = calculate_profit(y_test, y_prob, value_test, t)
    profits.append(p)

# Find optimal
max_profit = max(profits)
best_threshold = thresholds[np.argmax(profits)]

print(f"Maximum Profit: ${max_profit:,.2f} at Threshold: {best_threshold:.2f}")

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.plot(thresholds, profits, label='XGBoost Profit Curve', color='blue', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal Threshold ({best_threshold:.2f})')
plt.axhline(0, color='gray', linestyle='-', linewidth=0.5)
plt.xlabel("Churn Probability Threshold")
plt.ylabel("Total Estimated Profit ($)")
plt.title("Profit-Driven Churn Analysis (XGBoost)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
