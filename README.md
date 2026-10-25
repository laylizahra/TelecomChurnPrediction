# 📉 Understanding Customer Characteristics and Predicting Churn in Telecommunications Using CatBoost

> **Who leaves a telecom company, and can we see it coming?**
> A tuned CatBoost model on the IBM Telco Customer Churn dataset that detects **72% of churners** and ranks customers into Low / Medium / High risk segments whose actual churn rises from **10% to 70%**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-blueviolet)
![Status](https://img.shields.io/badge/Status-Course%20Project-green)

*Small Project for Introduction to Machine Learning, Institut Teknologi Sepuluh Nopember (ITS), Surabaya.*

> 📎 **All charts and diagrams (method flow, confusion matrix, feature importance, risk segmentation, overfitting check) are in the presentation file: [`docs/MIDTERM_MACHINE_LEARNING.pptx`](docs/MIDTERM_MACHINE_LEARNING.pptx).** This README summarizes the results in text and tables.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Findings](#-key-findings)
3. [Results at a Glance](#-results-at-a-glance)
4. [Dataset](#-dataset)
5. [Methodology](#-methodology)
6. [Detailed Results](#-detailed-results)
7. [Business Recommendations](#-business-recommendations)
8. [Limitations](#-limitations)
9. [Quick Start](#-quick-start)
10. [Repository Structure](#-repository-structure)
11. [Team](#-team)
12. [References](#-references)

---

## 🎯 Project Overview

In our dataset, **1 of 4 customers left the company** (1,869 of 7,043). Winning a new customer costs more than keeping an existing one, so a company that can predict churn *before* it happens can act and avoid profit loss.

This project has three goals:

1. **Analyze customer characteristics** and identify the key drivers of churn.
2. **Develop and optimize** a machine learning model to predict churn.
3. **Evaluate performance** and compare it against a baseline model (Logistic Regression).

### Why CatBoost?
- 16 of the 19 original features are categorical (contract type, payment method, etc.), and CatBoost handles categorical data natively without heavy preprocessing.
- Gradient boosting builds trees one by one, each correcting the errors of the previous ones, which gives accurate and stable predictions.
- Class imbalance (26.5% churn) is handled with **threshold tuning** instead of resampling.

### Challenges and our ideas

| Challenge | Our idea |
|---|---|
| Manual hyperparameter tuning is inefficient | Use **Optuna** to optimize hyperparameters |
| Many categorical features | Use CatBoost's **native categorical handling** |
| Raw features may not capture churn patterns | Create new features through **feature engineering** |

---

## 🔑 Key Findings

- **Churn concentrates among new customers on month-to-month contracts with weak service engagement.**
- **Contract type** and **tenure** are the dominant churn drivers. Charges, support services, internet service, and payment method contribute less.
- Customers in their **first 6 months churn at 52.9%**, twice the average.
- **Two-year contracts churn at only 2.8%.**
- Churners stay about **half as long** (18.0 vs 37.6 months on average) and pay about **22% more per month** ($74.44 vs $61.27).
- The model's risk segments separate customers well: actual churn rises from **10% (Low)** to **70% (High)**.

---

## 📊 Results at a Glance

Evaluated on a **held-out test set of 1,409 customers** (374 churners), which was never used for tuning or threshold selection.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.7757 | 0.5607 | 0.7166 | 0.6291 |
| **CatBoost (ensemble)** | **0.7821** | **0.5711** | **0.7193** | **0.6367** |

> 💡 Because the data is imbalanced, **recall and F1 for the churn class are more informative than overall accuracy.**

📎 *See the **Discussion** slide in the presentation for the risk segmentation chart.*

---

## 🗂️ Dataset

| Item | Detail |
|---|---|
| Name | IBM Telco Customer Churn |
| Size | 7,043 rows × 21 columns |
| Target | `Churn` (Yes/No → 1/0) |
| Stayed | 5,174 customers (73.5%) |
| Churned | 1,869 customers (26.5%) |
| Duplicates | 0 |
| Missing values | 11 blank `TotalCharges` (all new customers with tenure = 0, never billed) |

### Numerical profile

| Aspect | Stayed | Churned | Difference |
|---|---|---|---|
| Tenure (mean) | 37.6 months | 18.0 months | −52%, churners leave early |
| Monthly charges (mean) | $61.27 | $74.44 | +22%, churners pay more per month |
| Total charges (mean) | $2,549.91 | $1,531.80 | −40%, less accumulated spending |

### Highest-risk segments (EDA)

| Aspect | Category | Churn rate | Lift vs. average |
|---|---|---|---|
| Tenure group | 0–6 months | 52.9% | 2.0× |
| Payment method | Electronic check | 45.3% | 1.7× |
| Contract | Month-to-month | 42.7% | 1.6× |
| Internet service | Fiber optic | 41.9% | 1.6× |
| Tech support | No | 41.6% | 1.6× |
| *Lowest: Contract* | *Two year* | *2.8%* | *0.1×* |

---

## 🛠️ Methodology

📎 *See the **Proposed Method** slide in the presentation for the full pipeline diagram.*

### 1. Cleaning
- `TotalCharges` converted from text to numeric; the 11 blank values (tenure = 0, never billed) filled with `0`.
- `SeniorCitizen` recoded 0/1 → No/Yes; `Churn` recoded Yes/No → 1/0.
- `CustomerID` removed (unique identifier, not a feature).
- No other missing values and no duplicate customer IDs.

### 2. Feature engineering (19 → 44 features)
25 new features were created **before the split, from input columns only (no target used)**:

| Group | # | Description |
|---|---|---|
| Service counts | 6 | Number of add-on services |
| Billing behavior | 7 | e.g., average monthly charge |
| Customer lifecycle | 4 | e.g., new-customer flag |
| Risk profile from EDA | 6 | Risk flags derived from EDA findings |
| Categorical interactions | 2 | e.g., contract × payment method |

### 3. Encoding

| Model | Input |
|---|---|
| **CatBoost** (main) | 44 features: 19 categorical + 25 numeric. Categoricals used as they are: no one-hot, no scaling. |
| **Logistic Regression** (baseline) | One-hot encoding + standard scaling (94 columns), fitted on training folds only inside a pipeline. |

### 4. Data split

| Set | Size | Churn |
|---|---|---|
| Development (80%) | 5,634 rows | 1,495 (26.5%) |
| Test (20%) | 1,409 rows | 374 (26.5%) |

The test set is used for **final evaluation only**.

### 5. Hyperparameter tuning with Optuna
- **Sampler:** TPE
- **Search space:** `learning_rate`, `depth`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`
- **Objective:** `CV AUC − overfitting penalty` (penalty = train–validation gap), so the search favors models that generalize.

### 6. Validation
- Stratified **5-fold CV** on the development set, with early stopping per fold.
- Overfitting check via the train vs. validation AUC gap.
- Learning curves (iterations and data size) computed on the development set, so the test set stays untouched.

### 7. Handling class imbalance
- **No oversampling and no class weights**, so predicted probabilities are not distorted.
- Imbalance is handled with **PR-AUC** and **threshold tuning**.
- The decision threshold is selected from **out-of-fold (OOF) predictions** (grid 0.05–0.79), using F1-optimal and cost-optimal criteria (FN : FP = 500 : 50), and compared with the default 0.50.

### 8. Models

**CatBoost** is an ensemble of trees, each fixing the errors of the previous ones:

```
F(x) = Σ η · h_t(x)        P(churn) = σ(F(x)) = 1 / (1 + e^(−F))
```
(η = learning rate, h_t = tree t)

**Logistic Regression** (baseline) is a single linear equation whose weights are feature effects:

```
z = wᵀx + b               P(churn) = σ(z) = 1 / (1 + e^(−z))
```

---

## 📈 Detailed Results

### Classification report (test set)

| Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| No churn | 0.888 | 0.805 | 0.844 | 1,035 |
| Churn | 0.571 | 0.719 | 0.637 | 374 |

### Confusion matrix (test set)

|  | Predicted No | Predicted Churn |
|---|---|---|
| **Actual No** | 833 | 202 |
| **Actual Churn** | 105 | 269 |

- ✅ The model detects **72% of churners** (269 of 374).
- ✅ When it predicts churn, it is correct **57% of the time**.
- ⚠️ **Missed churners (105):** their profile looks like a loyal customer, but the data lacks behavioral signals.
- ⚠️ **False alarms (202):** they match a high-risk profile but choose to stay.

📎 *See the **Evaluation** slide in the presentation for the confusion matrix plot.*

### Overfitting check (CV 5-fold)

| Metric | Train | Validation | Gap |
|---|---|---|---|
| ROC-AUC | 0.858 | 0.847 | 0.011 |
| Accuracy | 0.795 | 0.782 | 0.013 |
| Logloss | 0.402 | 0.414 | 0.012 |

The train–validation gap is very small on all metrics, which indicates the model is **not overfitting**.

📎 *See the **Overfitting Check** slide in the presentation for the logloss curve.*

### Main churn drivers (feature importance)

📎 *See the **Discussion** slide in the presentation for the feature importance chart.*

Contract type and tenure dominate; charges, support services, internet service, and payment method contribute less.

### Risk segmentation (test set)

| Risk segment | n | Average predicted probability | Actual churn rate |
|---|---|---|---|
| Low | 883 | 0.101 | 0.100 |
| Medium | 321 | 0.444 | 0.445 |
| High | 205 | 0.711 | 0.698 |

Predicted probabilities closely match actual churn rates, so the segments are well separated and well calibrated.

### Customer profile

- **High-risk:** new customers (0–6 months, 53%), electronic check (45%), month-to-month contracts (43%), fiber optic (42%), no tech support (42%), senior citizens (42%).
- **Low-risk:** two-year contracts (3%) and long-tenure customers (10%).

---

## 🎯 Business Recommendations

1. **Encourage month-to-month customers to move to 1–2 year contracts** (e.g., with loyalty discounts).
2. **Develop automatic payment options** and evaluate the electronic check payment experience.
3. **Run onboarding programs for new customers** during their first 6 months.
4. **Prioritize the high-risk segment** in retention campaigns to use the budget where it matters most.

---

## ⚠️ Limitations

- **Modest gain over the baseline:** CatBoost improves on Logistic Regression only slightly (F1 0.6367 vs. 0.6291), which suggests the dataset's signal is largely linear.
- **Precision of 0.57:** roughly 1 in 2 churn alerts is a false alarm, so retention actions should be low-cost or well targeted.
- **No behavioral data:** the dataset contains demographic, account, and service information only (no usage, complaints, or interaction history), which explains many missed churners.
- **Single dataset:** results may not generalize to other telecom markets without re-validation.
- **Feature importance is associative, not causal:** it shows which variables the model uses, not that changing them will prevent churn.

---

**Main libraries:** `pandas`, `numpy`, `scikit-learn`, `catboost`, `optuna`, `matplotlib`, `seaborn`

> 📝 Replace the file names above with your actual notebook/script names.

---

## 📁 Repository Structure

> Adjust to match your actual files.

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_catboost.ipynb
├── docs/
│   └── MIDTERM_MACHINE_LEARNING.pptx   # full presentation with all charts
├── requirements.txt
└── README.md
```

---

## 👥 Team

**Small Project, Introduction to Machine Learning**
Department of Mathematics, Institut Teknologi Sepuluh Nopember (ITS), Surabaya

| Name | Role | GitHub |
|---|---|---|
| Haqi Ardika Ramadani | Team member | [@username](https://github.com/username) |
| Raychana Maharani Zahra | Team member | [@username](https://github.com/zahraychana) |
| Layli Nurul Fatimah Zahra | Team member | [@username](https://github.com/laylizahra) |

**Course Instructor:** Mohammad Iqbal, S.Si., M.Si., Ph.D.

---

## 📚 References

- Wagh, S. K., et al. (2024). *Results in Control and Optimization.* https://doi.org/10.1016/j.rico.2023.100342
- Asif, et al. (2025). *Results in Engineering.* https://doi.org/10.1016/j.rineng.2025.104629
- IBM Telco Customer Churn dataset.

---

## 🙏 Acknowledgements

Thanks to our course instructor for guidance throughout this project.

⭐ If you find this project useful, consider giving it a star!
