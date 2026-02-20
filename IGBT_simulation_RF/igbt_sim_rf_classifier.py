import os
import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load CSV files
thermal_df = pd.read_csv("igbt_thermal_rf_datasetv2.csv")
dynamic_df = pd.read_csv("IGBT_dynamic_rf_datasetv3.csv")
switching_df = pd.read_csv("IGBT_switching_loss_rf_datasetv3.csv")
#combined_df = pd.read_csv("")

# Strip whitespace from column names (very important)
thermal_df.columns = thermal_df.columns.str.strip()
dynamic_df.columns = dynamic_df.columns.str.strip()
switching_df.columns = switching_df.columns.str.strip()

labels = thermal_df["HealthState"]
thermal_df = thermal_df.drop(columns=["HealthState"])
dynamic_df = dynamic_df.drop(columns=["HealthState"])
switching_df = switching_df.drop(columns=["HealthState"])


# Combine all features into one dataset
X = pd.concat([thermal_df, dynamic_df, switching_df], axis=1)
y = labels

## Label Encoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# See encoding (for your report)
encoding_map = dict(zip(label_encoder.classes_,
                          label_encoder.transform(label_encoder.classes_)))
print("Label encoding:", encoding_map)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Training the Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

print("\n================= MODEL PERFORMANCE =================")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
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

# Save the Model
#print("saving to", os.getcwd())
#joblib.dump(rf, "rf_igbt_fault_model.pkl")
#joblib.dump(label_encoder, "label_encoder.pkl")