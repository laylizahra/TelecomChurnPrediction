# 📉 Telco Customer Churn Prediction with CatBoost

> Who leaves a telecom company, and can we see it coming?
> A tuned CatBoost model that detects **72% of churners** and ranks
> customers into risk segments from **10% to 70%** actual churn.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-blueviolet)

## 🔑 Key Findings
- Churn concentrates among **new customers (0–6 months, 53% churn)** on
  **month-to-month contracts (43%)** paying by **electronic check (45%)**.
- **Contract type** and **tenure** are the strongest churn drivers.
- Two-year contracts churn only **2.8%**.

## 📊 Results (held-out test set, 1,409 customers)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.7757 | 0.5607 | 0.7166 | 0.6291 |
| **CatBoost** | **0.7821** | **0.5711** | **0.7193** | **0.6367** |

![Risk segmentation](images/risk_segmentation.png)

## 🧠 Why this project?
(1 in 4 customers left; retention is cheaper than acquisition)

## 🛠️ Approach
![Method](images/method.png)
1. Cleaning and EDA
2. Feature engineering: 19 → 44 features (created before the split, no target used)
3. CatBoost with native categorical handling
4. Optuna tuning with an overfitting penalty (5-fold CV)
5. Threshold chosen from out-of-fold predictions
6. Evaluation against a Logistic Regression baseline

## 💡 Design Decisions
- No SMOTE or class weights → probabilities stay calibrated
- Test set stays locked until the final evaluation
- ...

## ⚠️ Limitations
- Precision of 0.57 means about 1 in 2 alerts is a false alarm
- Only demographic and service data, with no behavioral signals

## 🎯 Business Recommendations
1. Encourage month-to-month customers to move to 1–2 year contracts
2. ...

## 🚀 Quick Start
git clone ... && pip install -r requirements.txt

## 📁 Repository Structure
## 👥 Team & Acknowledgements
Small project, Introduction to Machine Learning, ITS
## 📚 References
