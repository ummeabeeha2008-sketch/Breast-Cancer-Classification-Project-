"""
Breast Cancer Classification Project
--------------------------------------
Goal: Classify tumors as Malignant (cancerous) or Benign (non-cancerous)
based on cell measurements from biopsy data.

Dataset: Breast Cancer Wisconsin Dataset (built into scikit-learn)
"""

# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  # features (cell measurements)
y = data.target  # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)
print(X.head())

# Step 3: Split into training and testing sets
# 80% of data to train the model, 20% to test how well it learned
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale the features
# Many ML models perform better when features are on a similar scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a Logistic Regression model
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test, log_preds))
print(classification_report(y_test, log_preds, target_names=data.target_names))

# Step 6: Train a Random Forest model (often performs better, handles non-linear patterns)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaled data
rf_preds = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds, target_names=data.target_names))

# Step 7: Confusion Matrix (shows correct vs incorrect predictions)
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_preds))

# Step 8: Feature importance — which measurements matter most for diagnosis
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)
print("\nTop 10 most important features:")
print(top_features)
