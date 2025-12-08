import torch
import torch.nn as nn
import timm
from sentence_transformers import SentenceTransformer

class ImageEncoder(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window7_224', out_dim=512, pretrained=True):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self.backbone = backbone
        self.proj = nn.Linear(backbone.num_features, out_dim)
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat  # [B, out_dim]

class TabularEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.net(x)

class NotesEncoder:
    # wrapper around sentence-transformers (non-pt module for speed). returns numpy -> torch
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        self.model = SentenceTransformer(model_name)
        self.device = device
    def encode(self, texts):
        embs = self.model.encode(texts, convert_to_tensor=True, device=self.device)
        return embs  # torch.Tensor

class Aggregator(nn.Module):
    def __init__(self, im_dim, tab_dim, note_dim, hidden=512):
        super().__init__()
        total = im_dim + tab_dim + note_dim
        self.fusion = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2)
        )
    def forward(self, im, tab, note):
        x = torch.cat([im, tab, note], dim=1)
        return self.fusion(x)

class HeadClassifier(nn.Module):
    def __init__(self, in_dim, n_classes=2):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x): return self.cls(x)

class LRM(nn.Module):
    # produces Sc in [0,1]
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).squeeze(-1)
