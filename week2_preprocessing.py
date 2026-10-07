import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Data Collection Simulation
print("--- 1. Simulating Logistics Data ---")
np.random.seed(42)
df = pd.DataFrame({
    'Shipment_ID': range(1000, 2000),
    'Distance_km': np.random.uniform(10, 500, 1000),
    'Weight_kg': np.random.uniform(1, 100, 1000),
    'Transit_Time_Hours': np.random.uniform(1, 48, 1000)
})

# Introducing typical data issues (Missing values and Outliers)
df.loc[10:25, 'Transit_Time_Hours'] = np.nan
df.loc[5, 'Distance_km'] = 99999.0  # Massive outlier

print(f"Initial missing values:\n{df.isnull().sum()}")

# 2. Data Cleaning: Handling Missing Values
print("\n--- 2. Handling Missing Values ---")
# Using median imputation to avoid skewness
median_time = df['Transit_Time_Hours'].median()
df['Transit_Time_Hours'] = df['Transit_Time_Hours'].fillna(median_time)
print("Missing values after imputation:", df['Transit_Time_Hours'].isnull().sum())

# 3. Data Cleaning: Outlier Detection 
print("\n--- 3. Handling Outliers ---")
Q1 = df['Distance_km'].quantile(0.25)
Q3 = df['Distance_km'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping the outliers
df['Distance_km'] = np.clip(df['Distance_km'], lower_bound, upper_bound)
print(f"Max distance after capping outliers: {df['Distance_km'].max():.2f} km")

# 4. Data Preprocessing: Normalization
print("\n--- 4. Normalization ---")
scaler = MinMaxScaler()
numerical_cols = ['Distance_km', 'Weight_kg', 'Transit_Time_Hours']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Data preview after normalization (values between 0 and 1):")
print(df[numerical_cols].head())
