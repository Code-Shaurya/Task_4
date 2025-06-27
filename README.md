# Task_4
# Logistic Regression Binary Classifier

A Python project to train and evaluate a binary classification model using Logistic Regression. The code includes preprocessing, model training, evaluation, threshold tuning, and visualization.

---

## 🚀 Features

- Handles missing data via mean/mode imputation
- Standardizes numeric features
- Trains a logistic regression model
- Evaluates with:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1)
  - ROC Curve and AUC
- Visualizes the sigmoid activation function
- Allows threshold tuning for custom decision boundaries

---

## 📁 Dataset

Upload or place your dataset as `data.csv` in the root directory. The last column is assumed to be the binary target.

---

## 📦 Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn
