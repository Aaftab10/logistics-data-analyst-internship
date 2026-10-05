import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
n_shipments = 600

# 1. Simulate Logistics Operational Dataset
data = {
    'shipment_id': [f"SHP_{1000 + i}" for i in range(n_shipments)],
    'origin_hub': np.random.choice(['Hub_North', 'Hub_South', 'Hub_West'], n_shipments),
    'distance_km': np.random.uniform(10.0, 320.0, n_shipments).round(2),
    'traffic_congestion_index': np.random.uniform(1.0, 5.0, n_shipments).round(1),
    'cargo_weight_kg': np.random.uniform(5.0, 1800.0, n_shipments).round(1),
}
df = pd.DataFrame(data)

# Compute transit duration with noise and congestion impact
base_speed_kmh = 42.0
df['scheduled_hours'] = (df['distance_km'] / base_speed_kmh).round(2)
delay_factor = (df['traffic_congestion_index'] * 0.18) + (df['cargo_weight_kg'] / 10000.0)
df['actual_hours'] = (df['scheduled_hours'] * (1 + delay_factor) + np.random.normal(0.2, 0.1, n_shipments)).round(2)
df['delay_hours'] = (df['actual_hours'] - df['scheduled_hours']).clip(lower=0).round(2)
df['is_on_time'] = (df['delay_hours'] <= 0.25).astype(int)

# 2. KPI Evaluation
otd_rate = df['is_on_time'].mean() * 100
avg_distance = df['distance_km'].mean()
avg_delay = df['delay_hours'].mean()

print("--- WEEK 1 LOGISTICS KPI BASELINE ---")
print(f"Total Shipments Analyzed: {len(df)}")
print(f"On-Time Delivery Rate: {otd_rate:.2f}%")
print(f"Average Trip Distance: {avg_distance:.2f} km")
print(f"Average Delay Duration: {avg_delay:.2f} hours")

# 3. Correlation Analysis
corr_matrix = df[['distance_km', 'traffic_congestion_index', 'cargo_weight_kg', 'delay_hours']].corr()
print("\nFeature Correlation with Delay:")
print(corr_matrix['delay_hours'])
