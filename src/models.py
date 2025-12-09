import torch
import torch.nn as nn
from torchvision.models import swin_base_patch4_window7_224
from transformers import AutoTokenizer, AutoModelForCausalLM

# ImageAgent: predicts independently from X-ray
class ImageAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = swin_base_patch4_window7_224(pretrained=True)
        self.backbone.head = nn.Identity()
        self.classifier = nn.Linear(1024,1)  # independent prediction

    def forward(self, x):
        feat = self.backbone(x)
        Cm = torch.sigmoid(self.classifier(feat))  # confidence score
        return feat, Cm

# TabularAgent: predicts independently from vitals/labs
class TabularAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self, x):
        Cm = torch.sigmoid(self.mlp(x))
        return x, Cm

# NotesAgent: predicts independently from clinical notes using LLM
class NotesAgent(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        self.head = nn.Linear(4096,1)  # project hidden state to confidence

    def forward(self, texts, device):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        note_feat = outputs.hidden_states[-1][:,0,:]
        Cm = torch.sigmoid(self.head(note_feat))
        return note_feat, Cm
