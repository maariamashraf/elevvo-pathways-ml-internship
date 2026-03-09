# Machine Learning Engineering Internship — Elevvo Pathways

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-green?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![Internship](https://img.shields.io/badge/Internship-Elevvo%20Pathways-purple?style=flat)

> A collection of 4 end-to-end machine learning projects completed during my ML Engineering 
> internship at Elevvo Pathways — spanning regression, binary classification, multi-class 
> classification, and unsupervised clustering.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [T1 — Student Score Prediction](#t1--student-score-prediction)
  - [T2 — Loan Approval Prediction](#t2--loan-approval-prediction)
  - [T3 — Customer Segmentation](#t3--customer-segmentation)
  - [T4 — Forest Cover Type Classification](#t4--forest-cover-type-classification)
- [Tech Stack](#tech-stack)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Author](#author)

---

## Overview

During my internship at **Elevvo Pathways**, I designed and delivered 4 machine learning 
projects from scratch — handling everything from raw data profiling and cleaning, through 
feature engineering and model selection, to final evaluation and benchmarking.

Each project tackles a distinct ML problem type and real-world domain:

| # | Project | ML Type | Dataset Size |
|---|---------|---------|-------------|
| T1 | Student Score Prediction | Regression | 6,607 rows |
| T2 | Loan Approval Prediction | Binary Classification | 614 rows |
| T3 | Customer Segmentation | Unsupervised Clustering | 200 rows |
| T4 | Forest Cover Classification | Multi-Class Classification | 581,012 rows |

---

## Projects

---

### T1 — Student Score Prediction

**Goal:** Predict a student's exam score based on behavioral, demographic, and 
academic factors.

**Dataset:** `StudentPerformanceFactors.csv` — 6,607 students · 20 features  
**Target:** `Exam_Score` (continuous)

**Approach:**
- Data profiling, null handling (mode imputation), IQR-based outlier clipping
- EDA: correlation heatmaps, distribution plots, score vs. feature line plots
- Preprocessing: StandardScaler + OneHotEncoder via ColumnTransformer
- Models: Linear Regression · Polynomial Regression (degree=2)
- Bonus: Feature selection — reduced to 10 most impactful features

**Results:**

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | **0.9899** | 0.3235 | 0.2705 |
| Polynomial Regression | 0.9887 | 0.3432 | — |
| Linear (10 features) | 0.8885 | 1.0773 | 0.8655 |

**Key Insight:** Linear relationships dominate — the full-feature linear model 
outperformed polynomial regression, achieving ~99% variance explanation.

---

### T2 — Loan Approval Prediction

**Goal:** Automatically predict whether a loan application should be approved 
or rejected based on applicant profile and financial history.

**Dataset:** `loan.csv` — 614 applications · 13 features  
**Target:** `Loan_Status` (Y / N) — Binary Classification

**Approach:**
- Null handling: mode imputation for categoricals, mean for numericals
- Feature engineering: log transforms (`LoanAmount_log`, `ApplicantIncome_log`),
  combined income (`totalincome`, `totalincome_log`)
- EDA: approval rate distribution, income & loan amount skewness, gender vs. approval
- Encoding: LabelEncoder for all categorical columns
- Imbalance handling: **SMOTE** — balanced training set from 154/337 → 337/337
- Scaling: StandardScaler post-SMOTE
- Models: SVM (RBF Kernel) · Decision Tree

**Results:**

| Model | Train Accuracy | Val Accuracy |
|-------|---------------|-------------|
| SVM (RBF) | 81.16% | 82.11% |
| **Decision Tree** | 79.97% | **85.37%** |

**Key Insight:** Decision Tree outperformed SVM with near-perfect recall on 
approved loans (only 1 false negative), making it the stronger production candidate.

---

### T3 — Customer Segmentation

**Goal:** Group mall customers into meaningful behavioral segments to enable 
targeted marketing and personalized business strategies.

**Dataset:** `Mall_Customers.csv` — 200 customers · 5 features  
**Features:** Gender · Age · Annual Income · Spending Score  

**Approach:**
- EDA: univariate (age, income, spending distributions), bivariate 
  (gender breakdowns), multivariate (income vs. spending scatter)
- Feature binning: age groups (Young/Adult/Senior), income groups 
  (Low/Medium/High/Very High)
- Preprocessing: StandardScaler · MinMaxScaler · LabelEncoder
- Optimal K: Elbow Method via KElbowVisualizer
- Models: K-Means · Hierarchical (Agglomerative) · DBSCAN

**Results:**

| Algorithm | Features | Silhouette Score |
|-----------|---------|-----------------|
| K-Means (k=5) | Income + Spending | Best visual separation |
| Hierarchical | Age + Income + Spending | 0.3955 |
| Hierarchical | + Gender (4D) | 0.3503 |
| DBSCAN | 4D features | Noise detection applied |

**5 Customer Segments Identified:**

-🔵 Average Customers — Moderate income and balanced spending behavior

-🟢 VIP Customers — High income and very high spending; premium target group

-🟡 Impulsive Spenders — Lower income but high spending, driven by trends and promotions

-🔴 Conservative Wealthy — High income but low spending; potential for upselling

-⚪ Budget Customers — Lower income and low spending; price-sensitive shoppers

---

### T4 — Forest Cover Type Classification

**Goal:** Predict the dominant forest cover type of a 30×30m land patch using 
cartographic variables — a large-scale, multi-class environmental classification problem.

**Dataset:** UCI ML Repository (ID: 31) — 581,012 records · 55 features  
**Target:** `Cover_Type` — 7 classes  

**Approach:**
- Profiling: 581K rows, 7 target classes, significant class imbalance 
  (Type 2: 48.8%, Type 4: 0.5%)
- Feature engineering: 3 new features created:
  - `total_distance_to_hydrology` — Euclidean distance from water
  - `hillshade_Mean` — average hillshade across 3 time points
  - `hillshade_Range` — max − min hillshade variation
- Stratified 80/20 split → 464,809 train / 116,203 test
- Scaling: StandardScaler on all continuous columns
- Models: Random Forest · LightGBM · Gradient Boosting

**Results:**

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **Random Forest** | **94.79%** | 0.91 |
| LightGBM | 94.76% | **0.93** |
| Gradient Boosting | 89.63% | 0.87 |

**Key Insight:** Random Forest led on overall accuracy while LightGBM achieved 
better balanced recall on minority classes — making it the preferred choice for 
imbalanced real-world deployment.

---

## Tech Stack

**Core**
```
Python · Pandas · NumPy · Google Colab
```

**Machine Learning**
```
Scikit-learn · LightGBM · imbalanced-learn (SMOTE)
Linear & Polynomial Regression · SVM · Decision Tree
Random Forest · Gradient Boosting · Logistic Regression
K-Means · DBSCAN · Agglomerative Clustering
```

**Preprocessing**
```
StandardScaler · MinMaxScaler · LabelEncoder
OneHotEncoder · ColumnTransformer · Pipeline
```

**Visualization**
```
Matplotlib · Seaborn · Yellowbrick
```

**Evaluation**
```
MAE · RMSE · R² · Accuracy · F1-Score
Silhouette Score · Confusion Matrix · ROC-AUC
Calinski-Harabasz Score · Davies-Bouldin Score
```

---

## Results Summary

| Project | Algorithm | Best Metric |
|---------|-----------|-------------|
| Student Score Prediction | Linear Regression | R² = **98.99%** |
| Loan Approval Prediction | Decision Tree | Accuracy = **85.37%** |
| Customer Segmentation | K-Means (k=5) | Silhouette = **0.396** |
| Forest Cover Classification | Random Forest | Accuracy = **94.79%** |

---

## Repository Structure
```
 elevvo-pathways-ml-internship/
├── T1_Student_Score_prediction.ipynb
├── T2_Loan_Approval_Prediction.ipynb
├── T3_Customer_Segmentation.ipynb
├── T4_forest_cover_types_classification.ipynb
└── README.md
```

---

##  How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/elevvo-pathways-ml-internship.git
cd elevvo-pathways-ml-internship
```

2. **Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm imbalanced-learn yellowbrick ucimlrepo
```

3. **Open any notebook**
```bash
jupyter notebook
```
Or upload directly to **Google Colab** for a zero-setup experience.

4. **Dataset notes**
- T1, T2, T3: CSV files (update the file path in the notebook to your local path)
- T4: Fetched automatically from UCI ML Repository via `ucimlrepo`

---

## Author

**[Mariam Ashraf Hashim]**  
ML Engineering Intern · Elevvo Pathways  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/mariam-ashraf-069561317/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/maariamashraf)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:mariammashraf21@gmail.com)

---

> *"I don't just build models — I build solutions."*
