#!/usr/bin/env bash
# Minimal preprocessing script for clinical.csv and image paths
# Assumes you have downloaded MIMIC data and placed images in data/images.

set -euo pipefail

CSV_IN="data/raw_clinical.csv"
CSV_OUT="data/clinical.csv"
IMAGE_DIR="data/images"

if [ ! -f "$CSV_IN" ]; then
  echo "Place your raw clinical CSV at $CSV_IN"
  exit 1
fi

# Example transformations: keep columns patient_id, image_path, note_text, label, and numeric labs
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/raw_clinical.csv")
# ensure required columns exist. This is a minimal placeholder pipeline:
required = ['patient_id','image_path','note_text','label']
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in raw CSV: {missing}")
# keep numeric columns as tabular features
numeric = df.select_dtypes(include=['number']).columns.tolist()
tabular_cols = [c for c in numeric if c not in ['label']]
cols = ['patient_id','image_path','note_text','label'] + tabular_cols
df = df[cols]
df.to_csv("data/clinical.csv", index=False)
print("Wrote data/clinical.csv with columns:", list(df.columns))
PY

echo "Preprocessing done. clinical.csv created."
