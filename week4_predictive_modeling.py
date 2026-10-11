import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Problem Definition & Data Simulation
print("--- 1. Simulating Logistics Dataset ---")
np.random.seed(42)
n_samples = 2000
df = pd.DataFrame({
    'Distance_km': np.random.uniform(20, 800, n_samples),
    'Volume_kg': np.random.uniform(5, 1000, n_samples),
    'Traffic_Index': np.random.uniform(1, 10, n_samples)
})

# Target Variable: Forecasting Delivery Time (with non-linear complexity)
df['Delivery_Time_hrs'] = (df['Distance_km'] / 55) * (1 + (df['Traffic_Index'] / 20)) + (df['Volume_kg'] / 500) + np.random.normal(0, 2, n_samples)

# 2. Data Preparation for Modeling
X = df[['Distance_km', 'Volume_kg', 'Traffic_Index']]
y = df['Delivery_Time_hrs']

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Selection and Training
print("\n--- 2. Training Random Forest Regressor ---")
# Choosing an ensemble method (Random Forest) to handle non-linear traffic and volume effects
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation and Validation
print("\n--- 3. Evaluation Metrics ---")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} hours")
print(f"R-squared (R2): {r2:.4f}")

# 5. Optimization Strategies Extraction
print("\n--- 4. Feature Importance for Optimization ---")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print(feature_importances.sort_values(ascending=False))
print("\nOptimization Recommendation: Distance and Traffic Index are the heaviest predictors of delay. Dynamic route reallocation based on live traffic feeds will minimize RMSE and cost.")
