# 🚚 Logistics Data Analyst Internship

Welcome to my repository for the **Logistics Data Analyst Internship**. This project simulates real-world supply chain and logistics scenarios, focusing on data-driven decision-making, route optimization, and operational bottleneck identification.

## 👨‍💻 Author
**Ansari Aaftab Husain**

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning & Preprocessing:** Scikit-Learn

## 📂 Project Structure & Weekly Tasks

### 📌 Week 1: Strategic Planning and Data Exploration
* **Objective:** Define a logistics scenario, identify Key Performance Indicators (KPIs) like On-Time Delivery Rate (OTD) and Cost Per Ton-Mile (CPTM).
* **Output:** Created a baseline analytical roadmap and simulated initial logistics data.
* **File:** `week1_task.py`

### 📌 Week 2: Data Collection, Cleaning, and Preprocessing
* **Objective:** Build a robust data preprocessing pipeline.
* **Key Steps:** 
  * Handled missing timestamps using median imputation.
  * Detected and capped extreme distance anomalies using the IQR method.
  * Applied Min-Max Normalization for distance-based algorithm readiness.
* **File:** `week2_preprocessing.py`

### 📌 Week 3: Advanced Data Analysis and Visualization
* **Objective:** Uncover hidden operational bottlenecks through Exploratory Data Analysis (EDA).
* **Key Visualizations:**
  * Histogram of delivery times showing right-skewed delay patterns.
  * Scatter plots analyzing transportation costs against shipment volume.
  * Correlation heatmaps identifying the relationship between package weight and transit delays.
* **File:** `week3_visualization.py`

### 📌 Week 4: Predictive Modeling and Optimization
* **Objective:** Forecast delivery times using machine learning and propose operational optimization strategies.
* **Key Steps:** 
  * Simulated logistical data with non-linear traffic and volume effects.
  * Implemented and trained a **Random Forest Regressor** (Ensemble Method) for high-accuracy predictions.
  * Evaluated performance using MAE, RMSE, and R-squared metrics.
  * Recommended dynamic route planning and load-balancing based on feature importance insights.
* **File:** `week4_predictive_modeling.py`

## 🚀 How to Run
To run the scripts locally, clone this repository and install the required dependencies:

```bash
git clone [https://github.com/Aaftab10/logistics-data-analyst-internship.git](https://github.com/Aaftab10/logistics-data-analyst-internship.git)
cd logistics-data-analyst-internship
pip install pandas numpy matplotlib seaborn scikit-learn
