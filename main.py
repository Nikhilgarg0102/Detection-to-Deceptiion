# ----------------------------
# Gotham ML Cybersecurity Project
# ----------------------------

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# STEP 1: Load Dataset
# ----------------------------
dataset_dir = r"E:\XYZ\gotham_dataset\processed"  # Update path to your dataset
file_name = "iotsim-air-quality-1.csv"
file_path = os.path.join(dataset_dir, file_name)

# Load CSV
df = pd.read_csv(file_path, low_memory=False)
print("✅ Dataset Loaded")
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

# ----------------------------
# STEP 2: Preprocess Data
# ----------------------------
# Drop unusable columns
drop_cols = ['frame.time', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst']
df = df.drop(columns=drop_cols, errors='ignore')

# Define label
label_column = "label"
y = df[label_column]

# Encode categorical columns in features
X = df.drop(columns=[label_column])
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Encode labels
le_label = LabelEncoder()
y = le_label.fit_transform(y)
print("Classes:", le_label.classes_)

# Fill missing values
X = X.fillna(0)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# STEP 3: Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)

# ----------------------------
# STEP 4: Train Random Forest
# ----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ----------------------------
# STEP 5: Evaluate Model
# ----------------------------
y_pred = clf.predict(X_test)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_label.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=le_label.classes_, yticklabels=le_label.classes_)
plt.title("Confusion Matrix - Gotham Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------
# STEP 6: Feature Importance
# ----------------------------
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15 features
plt.figure(figsize=(10, 6))
plt.bar(range(len(indices)), importances[indices], align="center")
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=45, ha="right")
plt.title("Top 15 Important Features for Attack Detection")
plt.show()

# ----------------------------
# STEP 7: Save the Model (Optional)
# ----------------------------
import joblib
joblib.dump(clf, "gotham_rf_model.pkl")
print("✅ Model saved as 'gotham_rf_model.pkl'")
