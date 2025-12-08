#!/usr/bin/env bash
set -e

CSV_PATH=${1:-data/clinical.csv}
IMG_ROOT=${2:-data/images}
OUT_CSV=${3:-data/clinical_clean.csv}

echo "[*] Preprocess MIMIC-lite"
echo "  csv: $CSV_PATH"
echo "  images: $IMG_ROOT"
echo "  out: $OUT_CSV"

python - <<'PY'
import pandas as pd, os, sys
csv = os.environ.get('CSV_PATH', '${CSV_PATH}')
img_root = os.environ.get('IMG_ROOT', '${IMG_ROOT}')
out_csv = os.environ.get('OUT_CSV', '${OUT_CSV}')

df = pd.read_csv(csv)
print(f"[+] Loaded {len(df)} rows")

# Ensure required columns exist; if not, create placeholders (user should edit)
required = ['patient_id','image_path','note_text','label','age','heart_rate','systolic_bp']
for c in required:
    if c not in df.columns:
        print(f"[!] Column {c} missing — creating default zeros/empty")
        if c == 'label':
            df[c] = 0
        elif c == 'note_text':
            df[c] = ''
        else:
            df[c] = 0

# Fix image paths: check existence; drop rows with missing images
valid_idx = []
for idx, row in df.iterrows():
    p = os.path.join(img_root, str(row['image_path']))
    if os.path.exists(p):
        valid_idx.append(idx)
    else:
        # try if image_path already absolute
        if os.path.exists(str(row['image_path'])):
            df.at[idx, 'image_path'] = str(row['image_path'])
            valid_idx.append(idx)
        else:
            # drop or keep? we'll drop for safety
            pass

print(f"[+] {len(valid_idx)} valid image rows found out of {len(df)}")
df = df.loc[valid_idx].reset_index(drop=True)

# Fill NaNs
df['note_text'] = df['note_text'].fillna('')
df[['age','heart_rate','systolic_bp']] = df[['age','heart_rate','systolic_bp']].fillna(0)

df.to_csv(out_csv, index=False)
print(f"[+] Cleaned CSV written to {out_csv}")
PY
