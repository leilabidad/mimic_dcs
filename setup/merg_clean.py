import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

DATA_DIR = "/media/mohammad/Vir3/NLP_data_img_ipg/physionet.org/files/mimic-cxr-jpg/2.1.0"
FILES_DIR = os.path.join(DATA_DIR, "files")
CHEXPERT_CSV = os.path.join(DATA_DIR, "mimic-cxr-2.0.0-chexpert.csv.gz")
NEGBIO_CSV = os.path.join(DATA_DIR, "mimic-cxr-2.0.0-negbio.csv.gz")
METADATA_CSV = os.path.join(DATA_DIR, "mimic-cxr-2.0.0-metadata.csv.gz")
SPLIT_CSV = os.path.join(DATA_DIR, "mimic-cxr-2.0.0-split.csv.gz")
TEST_LABELED_CSV = os.path.join(DATA_DIR, "mimic-cxr-2.1.0-test-set-labeled.csv")

metadata = pd.read_csv(METADATA_CSV, compression='gzip')
split = pd.read_csv(SPLIT_CSV, compression='gzip')
chexpert = pd.read_csv(CHEXPERT_CSV, compression='gzip')
negbio = pd.read_csv(NEGBIO_CSV, compression='gzip')
test_labels = pd.read_csv(TEST_LABELED_CSV)

df = metadata.merge(split, on='dicom_id', how='left')
study_col = [c for c in df.columns if 'study' in c.lower()][0]
df = df.merge(chexpert, left_on=study_col, right_on='study_id', how='left', suffixes=('', '_chexpert'))
df = df.merge(negbio, left_on=study_col, right_on='study_id', how='left', suffixes=('', '_negbio'))

label_cols = [c for c in df.columns if c in [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum",
    "Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion",
    "Pleural Other","Pneumonia","Pneumothorax","Support Devices",
    "Atelectasis_negbio","Cardiomegaly_negbio","Consolidation_negbio","Edema_negbio",
    "Enlarged Cardiomediastinum_negbio","Fracture_negbio","Lung Lesion_negbio",
    "Lung Opacity_negbio","No Finding_negbio","Pleural Effusion_negbio",
    "Pleural Other_negbio","Pneumonia_negbio","Pneumothorax_negbio","Support Devices_negbio"
]]

def clean_labels(col):
    col = col.fillna(0)
    col = col.replace(-1, 0.5)
    return col.astype(float)

for col in label_cols:
    df[col] = clean_labels(df[col])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])

dicom_paths = {}
image_tensor_paths = []

for root, dirs, files in os.walk(FILES_DIR):
    jpg_files = [f for f in files if f.lower().endswith(".jpg")]
    for f in tqdm(jpg_files, desc="Processing images", unit="img"):
        dicom_id = os.path.splitext(f)[0]
        path = os.path.join(root, f)
        dicom_paths[dicom_id] = path
        try:
            img = Image.open(path).convert("L")
            tensor = transform(img).to(device)
            image_tensor_paths.append((dicom_id, path))
        except:
            continue

df["image_path"] = df["dicom_id"].map(dict(image_tensor_paths))
df = df[df["image_path"].notna()]
train_df = df[["dicom_id","image_path"] + label_cols]

FINAL_CSV = os.path.join(DATA_DIR, "mimic_cxr_numeric_ready.csv")
train_df.to_csv(FINAL_CSV, index=False)
print(f"✅ Numeric dataset ready: {FINAL_CSV}")
print(train_df.head())
print("Total images:", len(train_df))
