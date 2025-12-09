import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================
# 1. Preprocess MIMIC Dataset
# ==========================
class MIMICDataset(Dataset):
    def __init__(self, img_paths, tabular_df, notes_list, labels, tokenizer, transform=None):
        self.img_paths = img_paths
        self.tabular_df = tabular_df
        self.notes_list = notes_list
        self.labels = labels
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Image
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tabular
        tab_data = torch.tensor(self.tabular_df.iloc[idx].values, dtype=torch.float32)
        
        # Clinical Notes
        note_text = self.notes_list[idx]
        encoded_note = self.tokenizer(note_text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = encoded_note['input_ids'].squeeze(0)
        attention_mask = encoded_note['attention_mask'].squeeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, tab_data, input_ids, attention_mask, label

# Transform for images
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Sample dataset (باید با مسیرهای واقعی جایگزین شود)
img_paths = ["path/to/img1.png", "path/to/img2.png"]
labels = np.array([0,1])
tabular_df = pd.DataFrame({"age":[65,70],"heart_rate":[80,90]})
notes_list = ["Patient shortness of breath","No major complaints"]

train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

train_dataset = MIMICDataset([img_paths[i] for i in train_idx],
                              tabular_df.iloc[train_idx],
                              [notes_list[i] for i in train_idx],
                              labels[train_idx],
                              tokenizer,
                              transform=img_transform)

val_dataset = MIMICDataset([img_paths[i] for i in val_idx],
                            tabular_df.iloc[val_idx],
                            [notes_list[i] for i in val_idx],
                            labels[val_idx],
                            tokenizer,
                            transform=img_transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# ==========================
# 2. Setup Tabular & Notes Encoders
# ==========================
# Tabular Encoder
tab_encoder = nn.Sequential(
    nn.Linear(tabular_df.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 8)
).to(device)

# Clinical Notes Encoder (BioClinicalBERT)
note_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

# ==========================
# 3. Train/Fine-tune Base Models
# ==========================
# فرض: SwinNet از قبل آماده است و فقط TabEncoder و NoteEncoder آموزش داده می‌شوند
criterion = nn.BCEWithLogitsLoss()
optimizer_tab = optim.Adam(tab_encoder.parameters(), lr=1e-3)
optimizer_note = optim.Adam(note_model.parameters(), lr=1e-5)

for epoch in range(2):  # تعداد epochs نمونه
    tab_encoder.train()
    note_model.train()
    for img, tab, input_ids, attn_mask, label in train_loader:
        tab = tab.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        label = label.to(device)

        # Tabular forward
        tab_feat = tab_encoder(tab)

        # Notes forward
        outputs = note_model(input_ids=input_ids, attention_mask=attn_mask)
        note_feat = outputs.last_hidden_state[:,0,:]  # CLS token

        # Dummy loss برای نمونه: sum of features vs label
        loss_tab = criterion(tab_feat.mean(dim=1), label)
        loss_note = criterion(note_feat.mean(dim=1), label)
        loss = loss_tab + loss_note

        optimizer_tab.zero_grad()
        optimizer_note.zero_grad()
        loss.backward()
        optimizer_tab.step()
        optimizer_note.step()

    print(f"Epoch {epoch+1} completed, loss={loss.item():.4f}")

# ==========================
# 4. Train DCS Weights
# ==========================
Cm_list, Sc_list, y_val = [], [], []

tab_encoder.eval()
note_model.eval()

for img, tab, input_ids, attn_mask, label in val_loader:
    tab = tab.to(device)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    with torch.no_grad():
        # Cm: نمونه‌ای از VisionAgent (SwinNet) فرضی
        Cm = torch.rand(tab.shape[0]).cpu().numpy()
        # Sc: FusionAgent (می‌تواند جمع tab+note_feat باشد)
        tab_feat = tab_encoder(tab)
        note_feat = note_model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:,0,:]
        Sc = (tab_feat.mean(dim=1) + note_feat.mean(dim=1)).cpu().numpy() / 2

    Cm_list.extend(Cm)
    Sc_list.extend(Sc)
    y_val.extend(label.cpu().numpy())

# آموزش linear regression برای پیدا کردن w1, w2, w3
X = np.column_stack([Cm_list, Sc_list, np.array(Cm_list)*np.array(Sc_list)])
y_val = np.array(y_val)
reg = LinearRegression(fit_intercept=False).fit(X, y_val)
w1, w2, w3 = reg.coef_
print(f"Trained DCS weights: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")

# ==========================
# 5. Save Artifacts
# ==========================
os.makedirs("artifacts", exist_ok=True)
torch.save(tab_encoder.state_dict(), "artifacts/tab_encoder.pth")
note_model.save_pretrained("artifacts/note_encoder")
with open("artifacts/dcs_weights.pkl","wb") as f:
    pickle.dump({"w1":w1, "w2":w2, "w3":w3}, f)

print("Preprocessing, training, and DCS weight computation complete. Artifacts saved in 'artifacts/'")
