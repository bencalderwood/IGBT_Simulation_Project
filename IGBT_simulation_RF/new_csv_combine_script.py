import pandas as pd

# =========================
# Load CSV files
# =========================
thermal_df = pd.read_csv("igbt_thermal_rf_dataset.csv")
dynamic_df = pd.read_csv("IGBT_dynamic_rf_datasetv3.csv")
switching_df = pd.read_csv("IGBT_switching_loss_rf_datasetv3.csv")

#dynamic_df = dynamic_df.dropna()

# Option 2: fill with median
dynamic_df["Vge_plateau"] = dynamic_df["Vge_plateau"].fillna(dynamic_df["Vge_plateau"].median())

# =========================
# Define merge keys
# (adjust if you have more)
# =========================
merge_keys = ["Gate_voltage", "Temperature"]

# =========================
# Preserve ONE HealthState
# =========================
health_thermal = thermal_df[merge_keys + ["HealthState"]].copy()

# Drop HealthState from others to avoid duplication
dynamic_df = dynamic_df.drop(columns=["HealthState"])
switching_df = switching_df.drop(columns=["HealthState"])
thermal_df = thermal_df.drop(columns=["HealthState"])

# =========================
# Merge datasets
# =========================
merged_df = dynamic_df.merge(
    switching_df,
    on=merge_keys,
    how="inner"
)

merged_df = merged_df.merge(
    thermal_df,
    on=merge_keys,
    how="inner"
)

# =========================
# Reattach HealthState
# =========================
merged_df = merged_df.merge(
    health_thermal,
    on=merge_keys,
    how="inner"
)

# =========================
# Sanity checks
# =========================
print("Merged shape:", merged_df.shape)
print("Missing values per column:")
print(merged_df.isna().sum())

# Check label consistency
label_counts = merged_df["HealthState"].value_counts()
print("\nHealthState distribution:")
print(label_counts)

# =========================
# Save clean dataset
# =========================
merged_df.to_csv("IGBT_merged_clean3.csv", index=False)

print("\n✅ Clean ML-ready dataset saved as IGBT_merged_clean.csv")