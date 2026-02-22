# ====================================================
# HR ANALYTICS - ATTRITION DASHBOARD PROJECT
# Business KPI + EDA + Segmentation Analysis
# ====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# --------------------------------------------
# 2. Business KPIs
# --------------------------------------------
total_employees = len(df)
attrition_count = df[df["Attrition"] == "Yes"].shape[0]
attrition_rate = (attrition_count / total_employees) * 100
avg_income = df["MonthlyIncome"].mean()
avg_age = df["Age"].mean()

print("\n=== HR BUSINESS KPIs ===")
print(f"Total Employees  : {total_employees}")
print(f"Attrition Count  : {attrition_count}")
print(f"Attrition Rate % : {attrition_rate:.2f}")
print(f"Average Income   : {avg_income:.2f}")
print(f"Average Age      : {avg_age:.2f}")

# --------------------------------------------
# 3. Attrition Distribution
# --------------------------------------------
plt.figure()
sns.countplot(x="Attrition", data=df)
plt.title("Attrition Distribution")
plt.show()

# --------------------------------------------
# 4. Attrition by Department
# --------------------------------------------
plt.figure()
sns.countplot(x="Department", hue="Attrition", data=df)
plt.xticks(rotation=45)
plt.title("Attrition by Department")
plt.show()

# --------------------------------------------
# 5. Attrition by Job Role
# --------------------------------------------
plt.figure(figsize=(10,5))
sns.countplot(x="JobRole", hue="Attrition", data=df)
plt.xticks(rotation=90)
plt.title("Attrition by Job Role")
plt.show()

# --------------------------------------------
# 6. Monthly Income vs Attrition
# --------------------------------------------
plt.figure()
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income vs Attrition")
plt.show()

# --------------------------------------------
# 7. Years at Company vs Attrition
# --------------------------------------------
plt.figure()
sns.boxplot(x="Attrition", y="YearsAtCompany", data=df)
plt.title("Years at Company vs Attrition")
plt.show()

# --------------------------------------------
# 8. SQL-Style Business Analysis (Using Pandas)
# --------------------------------------------

print("\n=== Attrition by Department ===")
print(df.groupby("Department")["Attrition"].value_counts())

print("\n=== Average Income by Job Role ===")
print(df.groupby("JobRole")["MonthlyIncome"].mean().sort_values(ascending=False))

print("\n=== Attrition by Overtime ===")
print(df.groupby("OverTime")["Attrition"].value_counts())

print("\n=== Attrition by Marital Status ===")
print(df.groupby("MaritalStatus")["Attrition"].value_counts())

# --------------------------------------------
# 9. Correlation Heatmap (Numerical Factors)
# --------------------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True))
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# 10. Dashboard-Ready Aggregation Table
# --------------------------------------------
dashboard_df = df.groupby(["Department", "Attrition"]).agg(
    Total_Employees=("EmployeeNumber","count"),
    Avg_Income=("MonthlyIncome","mean"),
    Avg_Age=("Age","mean")
).reset_index()

print("\n=== Dashboard Summary Table ===")
print(dashboard_df.head())

# --------------------------------------------
# 11. Save Clean Dataset for Power BI
# --------------------------------------------
df.to_csv("HR_Analytics_Cleaned.csv", index=False)

print("\nClean dataset saved as HR_Analytics_Cleaned.csv")
