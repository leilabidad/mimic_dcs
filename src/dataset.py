"""
Dataset for MIMIC multi-modal experiments.
- Expects clinical.csv with columns:
  patient_id, image_path, note_text, label, [tabular features...]
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Optional

class MIMICDataset(Dataset):
    """PyTorch Dataset that returns a multimodal sample."""
    def __init__(self, csv_path: str, image_dir: str, image_size: int = 224,
                 tabular_cols: Optional[List[str]] = None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        # automatically detect tabular columns if not provided
        all_cols = list(self.df.columns)
        ignore = {'patient_id', 'image_path', 'note_text', 'label'}
        if tabular_cols is None or len(tabular_cols) == 0:
            self.tab_cols = [c for c in all_cols if c not in ignore]
        else:
            self.tab_cols = tabular_cols
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load image
        img_path = f"{self.image_dir}/{row['image_path']}"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # Tabular values (fillna with 0)
        tab = row[self.tab_cols].fillna(0.0).values.astype('float32')
        tab_tensor = torch.tensor(tab)
        # Note text
        note_text = str(row['note_text']) if not pd.isna(row['note_text']) else ""
        # Label
        label = float(row['label']) if not pd.isna(row['label']) else 0.0
        patient_id = row['patient_id']
        return {
            'image': image,
            'tabular': tab_tensor,
            'note_text': note_text,
            'label': torch.tensor(label, dtype=torch.float32),
            'patient_id': patient_id
        }
