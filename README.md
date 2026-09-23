# Credit Card Customer Churn Prediction

A machine learning project that predicts which bank customers are likely to
stop using their credit card ("churn"), so the bank can step in with
retention offers before they leave.

**Domain:** Data Science, Machine Learning &nbsp;|&nbsp; **Language:** Python &nbsp;|&nbsp; **Level:** Intermediate

## Overview

Banks lose revenue when credit card customers close their accounts. This
project trains a classifier on anonymized customer data to flag customers
at high risk of churning, so the bank can act early.

The pipeline:

1. **Load** the customer dataset from CSV.
2. **Clean** it — drop the ID column, fill missing categorical values.
3. **Encode & scale** — one-hot encode categorical columns (e.g. `Gender`,
   `Marital_Status`), standard-scale numerical columns.
4. **Balance classes with SMOTE** — churners are a small minority, so we
   synthesize extra churn examples for training.
5. **Train** a `RandomForestClassifier` on the balanced data.
6. **Evaluate** on held-out test data, reporting recall, a full
   classification report, and a confusion matrix — with recall on the
   "Attrited Customer" class as the key metric, since missing a churner is
   more costly than a false alarm.

## Project structure

```
churn-prediction/
├── churn_predictor.py     # main program: load, train, evaluate
├── generate_dataset.py    # creates a sample BankChurners.csv to run against
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

```bash
git clone <your-repo-url>
cd churn-prediction

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Get the dataset

This project expects a `BankChurners.csv` file in the project folder. You
have two options:

- **Use the real dataset:** download "Credit Card customers" from Kaggle
  and save it as `BankChurners.csv` in this folder.
- **Use the included generator:** run `python generate_dataset.py` to
  create a synthetic `BankChurners.csv` with the same columns, so you can
  try the project immediately without downloading anything.

### Run it

```bash
python churn_predictor.py
```

You'll see training progress followed by the evaluation output: the churn
recall score, a classification report, and a confusion matrix.

## Key technologies

- **Pandas / NumPy** — data loading and manipulation
- **Scikit-learn** — preprocessing, Random Forest, evaluation metrics
- **Imbalanced-learn** — SMOTE oversampling for the minority (churn) class

## Notes

- `generate_dataset.py` produces synthetic data for demonstration only —
  swap in the real Kaggle dataset for meaningful results.
- The Random Forest and SMOTE random states are fixed (`random_state=42`)
  so results are reproducible.
