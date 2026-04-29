import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/churn.csv")

# -----------------------------
# CLEANING
# -----------------------------
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# ENCODING
# -----------------------------
df = pd.get_dummies(df, drop_first=True)


# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ✅ SAVE FEATURE COLUMNS HERE (CORRECT PLACE)
import joblib
joblib.dump(X.columns, "models/feature_columns.pkl")

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/churn_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Load model
loaded_model = joblib.load("models/churn_model.pkl")
loaded_scaler = joblib.load("models/scaler.pkl")

# -----------------------------
# PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.show()

# Example new customer (same number of features as X)
sample = X.iloc[0:1]

# Scale
sample_scaled = loaded_scaler.transform(sample)

# Predict
prediction = loaded_model.predict(sample_scaled)

print("\nSample Prediction (0 = No Churn, 1 = Churn):", prediction)