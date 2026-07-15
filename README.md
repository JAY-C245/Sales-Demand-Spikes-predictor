# 🛒 Sales Demand Spike Predictor

A machine learning web app that predicts whether a product-day 
will experience a high-demand sales spike.

## 🔗 Live Demo
[👉 Click here to try the app](https://sales-demand-spikes-predictor-sme2e6zbdscmerevgsjjzy.streamlit.app)

## 🧠 How It Works
- Trained on 9,990 sales records with 98.2% / 1.8% class imbalance
- SMOTE applied to balance the minority class during training
- 4 models compared: Logistic Regression, Decision Tree, 
  Random Forest, Gradient Boosting
- Best model selected automatically by F1 Score

## ⚙️ Tech Stack
Python · scikit-learn · imbalanced-learn · Streamlit · GitHub

## 📊 Features Used
- Product & pricing data (unit price, discount, competitor price)
- Marketing & sentiment signals
- Lag & rolling revenue/units features
- Calendar & event flags (holidays, Black Friday, etc.)
