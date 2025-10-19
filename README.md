# Uber Ride Price Prediction and Dynamic Pricing

## Overview
Developed a machine learning model to predict Uber ride prices using XGBoost (R²=0.952) on Boston ride data (385K samples) and analyzed personal ride patterns from 2016. Includes dynamic pricing, holiday/airport trip analysis, and a geospatial heatmap.

## Key Features
- **EDA**: Analyzed trip categories, purposes, distances, hourly/weekly patterns, holiday surges, and airport trips.
- **Modeling**: XGBoost, Random Forest, Decision Tree, Linear Regression with RFE for feature selection.
- **Top Features**: Distance, cab type, source, destination, peak hours.
- **Dynamic Pricing**: Simulated surge pricing for peak hours, holidays, and weather, increasing average ride price by $X.
- **Visualizations**: Heatmap of Boston ride locations and plots for trip patterns.

## Tools
- Python (Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, Folium)
- SQL (ride aggregation)
- Tableau (dashboard: [link-to-your-Tableau-Public])

## Results
- XGBoost achieved R²=0.952, outperforming Linear Regression (R²=0.411).
- Identified key pricing factors (distance, cab type) via RFE.
- Dynamic pricing model suggests potential revenue increase.

## Setup
```bash
pip install -r requirements.txt
python uber_analysis.py
