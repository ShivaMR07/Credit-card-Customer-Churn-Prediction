"""
generate_dataset.py

Creates a synthetic sample of BankChurners.csv with the same columns as the
real Kaggle "Credit Card Customers" dataset, so the project runs end-to-end
without requiring an external download. Run once:

    python generate_dataset.py

This writes BankChurners.csv in the current folder.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 1000
CHURN_RATE = 0.16  # roughly matches the real dataset's class imbalance

attrition_flag = rng.choice(
    ["Existing Customer", "Attrited Customer"],
    size=N,
    p=[1 - CHURN_RATE, CHURN_RATE],
)
is_churn = attrition_flag == "Attrited Customer"

customer_age = rng.integers(26, 74, size=N)
gender = rng.choice(["M", "F"], size=N)
dependent_count = rng.integers(0, 6, size=N)
education_level = rng.choice(
    ["High School", "Graduate", "Uneducated", "College", "Post-Graduate", "Doctorate", "Unknown"],
    size=N,
)
marital_status = rng.choice(["Married", "Single", "Divorced", "Unknown"], size=N)
income_category = rng.choice(
    ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"],
    size=N,
)
card_category = rng.choice(["Blue", "Silver", "Gold", "Platinum"], size=N, p=[0.93, 0.05, 0.015, 0.005])

months_on_book = rng.integers(13, 57, size=N)
total_relationship_count = rng.integers(1, 7, size=N)

# Churned customers tend to have fewer months inactive contact patterns skewed higher
months_inactive_12_mon = np.where(
    is_churn,
    rng.integers(2, 5, size=N),
    rng.integers(0, 4, size=N),
)
contacts_count_12_mon = np.where(
    is_churn,
    rng.integers(2, 6, size=N),
    rng.integers(0, 4, size=N),
)

credit_limit = rng.uniform(1400, 34500, size=N).round(1)
total_revolving_bal = np.where(
    is_churn,
    rng.uniform(0, 500, size=N),
    rng.uniform(0, 2500, size=N),
).round(0)
avg_open_to_buy = (credit_limit - total_revolving_bal).clip(min=0).round(1)

total_amt_chng_q4_q1 = rng.uniform(0.3, 2.0, size=N).round(3)

total_trans_amt = np.where(
    is_churn,
    rng.uniform(500, 3000, size=N),
    rng.uniform(1500, 12000, size=N),
).round(0)
total_trans_ct = np.where(
    is_churn,
    rng.integers(10, 45, size=N),
    rng.integers(40, 130, size=N),
)
total_ct_chng_q4_q1 = rng.uniform(0.3, 2.0, size=N).round(3)
avg_utilization_ratio = np.where(
    credit_limit > 0, (total_revolving_bal / credit_limit).clip(0, 1), 0
).round(3)

df = pd.DataFrame(
    {
        "CLIENTNUM": rng.integers(700000000, 830000000, size=N),
        "Attrition_Flag": attrition_flag,
        "Customer_Age": customer_age,
        "Gender": gender,
        "Dependent_count": dependent_count,
        "Education_Level": education_level,
        "Marital_Status": marital_status,
        "Income_Category": income_category,
        "Card_Category": card_category,
        "Months_on_book": months_on_book,
        "Total_Relationship_Count": total_relationship_count,
        "Months_Inactive_12_mon": months_inactive_12_mon,
        "Contacts_Count_12_mon": contacts_count_12_mon,
        "Credit_Limit": credit_limit,
        "Total_Revolving_Bal": total_revolving_bal,
        "Avg_Open_To_Buy": avg_open_to_buy,
        "Total_Amt_Chng_Q4_Q1": total_amt_chng_q4_q1,
        "Total_Trans_Amt": total_trans_amt,
        "Total_Trans_Ct": total_trans_ct,
        "Total_Ct_Chng_Q4_Q1": total_ct_chng_q4_q1,
        "Avg_Utilization_Ratio": avg_utilization_ratio,
    }
)

df.drop_duplicates(subset="CLIENTNUM", inplace=True)
df.to_csv("BankChurners.csv", index=False)
print(f"Wrote BankChurners.csv with {len(df)} rows "
      f"({df['Attrition_Flag'].eq('Attrited Customer').mean():.1%} churned).")
