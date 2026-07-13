import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("../Dataset/train.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

# Remove Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

## Fill Missing Values

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# Fill categorical columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert numerical columns to numeric
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
# ==========================
# Label Encoding
# ==========================

label_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("\nMissing Values")
print(df.isnull().sum())
print("\nData Types After Encoding")
print(df.dtypes)

# Split Features & Target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
# ==========================
# Decision Tree Model
# ==========================

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)
# ==========================
# Random Forest Model
# ==========================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

# ==========================
# KNN Model
# ==========================

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", knn_accuracy)
# ==========================
# XGBoost Model
# ==========================

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBoost Accuracy:", xgb_accuracy)
# ==========================
# Save Best Model
# ==========================

joblib.dump(rf_model, "../Models/loan_model.pkl")

print("\nModel saved successfully!")