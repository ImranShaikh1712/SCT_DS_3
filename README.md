# Task 03 - Decision Tree Classifier

This project involves building a Decision Tree Classifier to predict whether a customer will purchase a product based on demographic and behavioral data using the Bank Marketing dataset from the UCI Machine Learning Repository.

## 📁 Dataset

- **Source**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Attributes**: Age, job, marital status, education, default, balance, housing, loan, etc.
- **Target**: `y` (yes/no – whether the client subscribed to a term deposit)

## 🧠 Model

- **Algorithm**: Decision Tree Classifier (`sklearn`)
- **Preprocessing**: Label Encoding of categorical variables
- **Evaluation**: Accuracy, Confusion Matrix, Classification Report

## 📊 Results

- Achieved good accuracy with entropy-based Decision Tree.
- Visualization of the decision-making process using `plot_tree`.

## 📦 Libraries Used

- `pandas`, `numpy`, `seaborn`, `matplotlib`
- `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.tree`, `sklearn.metrics`

## 🚀 How to Run

```bash
pip install pandas seaborn matplotlib scikit-learn
python decision_tree_bank.py
