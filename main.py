import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Telco_customer_churn.csv")

df["Total Charges"] = df["Total Charges"].str.strip()
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

y_true = df["Churn Value"]  

EXPECTED_MONTHS = 6
RETENTION_SUCCESS_RATE = 0.4
OFFER_COST = 60

df["Customer Value"] = df["Monthly Charges"] * EXPECTED_MONTHS

# Profit function
def calculate_profit(y_true, y_prob, customer_value, threshold):
    profit = 0

    for actual, prob, value in zip(y_true, y_prob, customer_value):
        offer = prob >= threshold

        if actual == 1 and offer:
            profit += (RETENTION_SUCCESS_RATE * value) - OFFER_COST
        elif actual == 1 and not offer:
            profit -= value
        elif actual == 0 and offer:
            profit -= OFFER_COST

    return profit

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

monthly_charge_score = normalize(df["Monthly Charges"])
tenure_score = 1 - normalize(df["Tenure Months"])

contract_risk = df["Contract"].map({
    "Month-to-month": 1.0,
    "One year": 0.4,
    "Two year": 0.1
})

df["Rule Churn Score"] = (
    0.5 * monthly_charge_score +
    0.3 * tenure_score +
    0.2 * contract_risk
)

df["Rule Churn Prob"] = normalize(df["Rule Churn Score"])

# Profit vs threshold
thresholds = np.linspace(0, 1, 50)
profits_rule = []

for t in thresholds:
    profits_rule.append(
        calculate_profit(
            y_true,
            df["Rule Churn Prob"],
            df["Customer Value"],
            t
        )
    )

# Plot
plt.figure()
plt.plot(thresholds, profits_rule, marker="o")
plt.xlabel("Churn Probability Threshold")
plt.ylabel("Total Profit")
plt.title("Profit vs Threshold — Rule-Based System")
plt.grid(True)
plt.show()

print("Profit range:", min(profits_rule), "to", max(profits_rule))
