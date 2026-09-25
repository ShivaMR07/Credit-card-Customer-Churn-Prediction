"""
churn_predictor.py

Credit Card Customer Churn Prediction
--------------------------------------
Loads a bank's customer dataset, cleans and encodes it, balances the classes
with SMOTE, trains a Random Forest classifier, and reports how well the
model identifies customers who are likely to churn.

Usage:
    python churn_predictor.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def load_and_prepare_data(filepath):
    """
    Loads the data, performs initial cleaning, and separates
    features (X) from the target variable (y).
    """
    # Load the dataset
    df = pd.read_csv(filepath)

    # Drop irrelevant columns. CLIENTNUM is just an ID, and the two
    # Naive Bayes columns (present in some versions of this dataset) are
    # leftovers from a prior analysis and aren't needed here.
    df = df.drop(
        columns=[
            "CLIENTNUM",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        ],
        errors="ignore",  # ignore if these columns aren't present
    )

    # Define the target variable 'y'. We want to predict 'Attrition_Flag'.
    # Convert the text labels to 0s and 1s for the model.
    df["Attrition_Flag"] = df["Attrition_Flag"].apply(
        lambda x: 1 if x == "Attrited Customer" else 0
    )
    y = df["Attrition_Flag"]

    # Define the features 'X' by dropping the target variable.
    X = df.drop("Attrition_Flag", axis=1)

    # Fill missing values in categorical columns with 'Unknown'
    for col in ["Education_Level", "Marital_Status", "Income_Category"]:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown")

    return X, y


def build_pipeline(X):
    """
    Builds the full preprocessing + SMOTE + model pipeline.
    """
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=["object", "string"]).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # For numerical data, scale it so no feature dominates due to its range.
    numerical_transformer = StandardScaler()

    # For categorical data, one-hot encode it. handle_unknown='ignore' means
    # categories unseen during training won't break prediction later.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Random Forest classifier, balanced to give extra weight to the
    # minority (churn) class.
    model = RandomForestClassifier(
        random_state=42, n_estimators=100, class_weight="balanced"
    )

    # Chain preprocessing -> SMOTE oversampling -> model training.
    full_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model),
        ]
    )

    return full_pipeline


def main():
    # Load the data
    X, y = load_and_prepare_data("BankChurners.csv")

    # Build the preprocessing + SMOTE + model pipeline
    full_pipeline = build_pipeline(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the model
    print("Training the model...")
    full_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    print("Evaluating the model...")
    y_pred = full_pipeline.predict(X_test)

    # Print the results
    print("\n--- Model Evaluation ---")
    # The recall for class '1' (Attrited Customer) is our key metric:
    # it tells us what fraction of actual churners the model catches.
    recall = recall_score(y_test, y_pred, pos_label=1)
    print(f"Recall for 'Attrited Customer': {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("------------------------")


if __name__ == "__main__":
    main()
