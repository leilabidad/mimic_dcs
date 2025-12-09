import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class MIMICDataset(Dataset):
    """Dataset for MIMIC multi-modal inputs: image, tabular, notes"""
    def __init__(self, csv_path, image_dir, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load image
        image = Image.open(f"{self.image_dir}/{row['image_path']}").convert('RGB')
        image = self.transform(image)
        # Tabular features
        tab_cols = [col for col in row.index if col not in ['image_path','note_text','label','patient_id']]
        tabular = torch.tensor([row[col] for col in tab_cols], dtype=torch.float)
        # Notes and labels
        note_text = row['note_text']
        label = torch.tensor(row['label'], dtype=torch.float)
        patient_id = row['patient_id']
        return {'image': image, 'tabular': tabular, 'note_text': note_text, 'label': label, 'patient_id': patient_id}
