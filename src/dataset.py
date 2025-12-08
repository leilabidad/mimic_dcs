import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import pandas as pd
from torchvision import transforms

class MIMICMultiModalDataset(Dataset):
    def __init__(self, csv_path, images_root, split='train', transform=None, max_text_len=512):
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.texts = self.df['note_text'].fillna('').tolist()
        self.tabular_cols = ['age','heart_rate','systolic_bp']  # نمونه — فیلدهای خودت رو بذار
        self.labels = self.df['label'].values.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        tabular = torch.tensor(row[self.tabular_cols].fillna(0).values.astype(float), dtype=torch.float32)
        text = row['note_text'] if isinstance(row['note_text'], str) else ''

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        meta = {'patient_id': row.get('patient_id', -1)}
        return {'image': image, 'tabular': tabular, 'text': text, 'label': label, 'meta': meta}
