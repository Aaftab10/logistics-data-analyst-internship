import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Simulation
np.random.seed(42)
print("Simulating Logistics Dataset...")
df = pd.DataFrame({
    'Distance_km': np.random.uniform(50, 1000, 1000),
    'Volume_kg': np.random.uniform(10, 500, 1000),
})

# Adding related metrics: Cost and Delivery Time
df['Cost_USD'] = (df['Distance_km'] * 1.2) + (df['Volume_kg'] * 0.8) + np.random.normal(0, 50, 1000)
df['Delivery_Time_hrs'] = (df['Distance_km'] / 60) + (df['Volume_kg'] / 150) + np.random.normal(0, 2, 1000)

# 2. Exploratory Data Analysis (EDA)
print("\n--- Exploratory Data Analysis (EDA) ---")
print(df.describe().round(2))

# 3. Visualizations

# Plot 1: Distribution of Delivery Times (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(df['Delivery_Time_hrs'], bins=30, kde=True, color='blue')
plt.title('Distribution of Delivery Times')
plt.xlabel('Delivery Time (Hours)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('delivery_time_distribution.png')
print("\nSaved: delivery_time_distribution.png")

# Plot 2: Transportation Cost vs. Volume (Scatter Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Volume_kg', y='Cost_USD', data=df, alpha=0.6, color='green')
plt.title('Transportation Cost vs. Shipment Volume')
plt.xlabel('Volume (kg)')
plt.ylabel('Cost (USD)')
plt.tight_layout()
plt.savefig('cost_vs_volume.png')
print("Saved: cost_vs_volume.png")

# Plot 3: Correlation Matrix (Heatmap)
plt.figure(figsize=(7, 5))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Logistics Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved: correlation_heatmap.png")

print("\nAnalysis Complete. Visualizations have been generated.")
