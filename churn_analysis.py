import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'https://raw.githubusercontent.com/swetha-624/customer-churn-analysis/main/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(url)

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert TotalCharges to numeric
df.dropna(inplace=True)  # Drop rows with missing values

# Overview of data
print("Dataset Overview:")
print(df.info())

# Distribution of target variable (Churn)
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Correlation Matrix
numeric_df = df.select_dtypes(include=[np.number])  # Select numeric columns
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Tenure vs Monthly Charges - Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df)
plt.title('Tenure vs Monthly Charges')
plt.xlabel('Tenure')
plt.ylabel('Monthly Charges')
plt.show()

# Boxplot for Monthly Charges by Churn
plt.figure(figsize=(8, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')
plt.show()

# Save processed data to a new CSV (optional)
processed_file = 'processed_churn_data.csv'
df.to_csv(processed_file, index=False)
print(f"Processed data saved as {processed_file}")
