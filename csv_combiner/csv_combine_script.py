import pandas as pd
import glob

all_files = [f for f in glob.glob("*.csv") if f != "IGBT_dynamic_5v_vge.csv"]

df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
df.to_csv("IGBT_combined_rf_datasetv3.csv", index=False)

print("All CSV files combined safely!")