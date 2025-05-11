import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_index_by_patient(index_csv, output_dir, seed=42, train_ratio=0.7, val_ratio=0.15):
    df = pd.read_csv(index_csv)

    # Filter out unknown labels (optional but recommended)
    df = df[df["label"] != "Unknown"].reset_index(drop=True)

    # Unique patient IDs
    patient_ids = df["patient_id"].unique()
    patient_ids.sort()

    # Split patient IDs
    train_ids, temp_ids = train_test_split(patient_ids, train_size=train_ratio, random_state=seed, shuffle=True)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)

    # Assign sets
    df_train = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    df_val   = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    df_test  = df[df["patient_id"].isin(test_ids)].reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    # Save to CSVs
    df_train.to_csv(os.path.join(output_dir, "index_train.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "index_val.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "index_test.csv"), index=False)

    print(f"✅ Train: {len(train_ids)} patients, {len(df_train)} segments")
    print(f"✅ Val:   {len(val_ids)} patients, {len(df_val)} segments")
    print(f"✅ Test:  {len(test_ids)} patients, {len(df_test)} segments")
    print(f"📦 Splits saved to: {output_dir}")


if __name__ == "__main__":
    split_index_by_patient(
        index_csv="features/index.csv",
        output_dir="features/splits/",
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15
    )
