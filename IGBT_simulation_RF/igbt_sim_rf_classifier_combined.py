import os
import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load combined dataset
df = pd.read_csv("IGBT_combined_rf_datasetv3.csv")

# Strip whitespace (good practice)
df.columns = df.columns.str.strip()

# Separate features and label
y = df["HealthState"]
X = df.drop(columns=["HealthState"])

# Encode labels
label_map = {
    "Healthy": 0,
    "Warning": 1,
    "Fault": 2
}

le = LabelEncoder()
le.classes_ = np.array(["Healthy", "Warning", "Fault"])
y_encoded = le.transform(y)
encoding_map = dict(zip(le.classes_,
                          le.transform(le.classes_)))
print("Label encoding:", encoding_map)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Train RF
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Simulation RF – Confusion Matrix")
plt.show()

# Feature importance

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n================= FEATURE IMPORTANCE =================")
print(importance_df)


plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.grid(True)
plt.savefig("igbt_rf_feature_importance.png", dpi=300)
plt.show()