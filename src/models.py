"""
Model agents:
 - VisionAgent: independent vision model (Swin) with independent prediction head
 - LabAgent: MLP for tabular data with independent prediction head
 - NotesAgent: LLM-based agent that produces embedding + text-based risk score (Cm)
All agents return: (features, Cm, structured_findings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_base_patch4_window7_224
from typing import List, Tuple, Dict, Any

# Transformers imports for notes agent
from transformers import AutoTokenizer, AutoModelForCausalLM

class VisionAgent(nn.Module):
    """Vision agent that predicts from chest x-ray and returns features + local confidence."""
    def __init__(self, backbone_name: str = "swin_base_patch4_window7_224", pretrained: bool = True):
        super().__init__()
        # Using torchvision swin base. It yields a 1024-dim feature by default
        self.backbone = swin_base_patch4_window7_224(pretrained=pretrained)
        # remove classification head; keep embedding
        self.backbone.head = nn.Identity()
        # small classification head on top (independent)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
          feat: [B,1024] image feature
          Cm:  [B,1] confidence (sigmoid output)
          findings: dict of structured findings for this agent (for audit)
        """
        feat = self.backbone(images)  # [B,1024]
        logit = self.classifier(feat)  # [B,1]
        Cm = torch.sigmoid(logit)
        # simple structured findings: normalized top-3 activations (example placeholder)
        # In production you would have a detection/segmentation head producing real findings
        topk_vals, _ = torch.topk(torch.abs(feat), k=3, dim=1)
        findings = {'top_feat_magnitudes': topk_vals.detach().cpu().tolist()}
        return feat, Cm, findings

class LabAgent(nn.Module):
    """Lab/Tabular agent: MLP that returns feature vector and confidence."""
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden//2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, tabular: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        feat = self.mlp(tabular)
        logit = self.classifier(feat)
        Cm = torch.sigmoid(logit)
        # produce simple anomaly scores (z-score proxy)
        anomaly_score = torch.tanh(torch.mean(feat, dim=1, keepdim=True))
        findings = {'anomaly_score': anomaly_score.detach().cpu().tolist()}
        return feat, Cm, findings

class NotesAgent:
    """
    Notes agent implemented against a causal LLM. This class wraps tokenization and lightweight
    projection of hidden states to a confidence score. The model may be large; we keep it as a
    non-torch nn.Module for flexibility (it uses transformers).
    """
    def __init__(self, model_name: str, device: str = 'cuda', quantized: bool = False):
        # This object intentionally does not inherit from nn.Module because the LLM may be large
        self.model_name = model_name
        self.device = device
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Model: AutoModelForCausalLM (we will access hidden states)
        # Use device_map="auto" to let transformers place layers across available devices if accelerate is configured
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        # Small projection head to map LLM last hidden state to a scalar confidence
        last_hidden_size = self.model.config.hidden_size
        # Build a small torch head on CPU or GPU depending on availability
        self.head = nn.Sequential(
            nn.Linear(last_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

    def encode_notes(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Tokenizes and runs the LLM to get last hidden state CLS token embedding and a confidence.
        Returns:
          note_feat: [B, hidden_size] (torch tensor on device of head)
          Cm_note: [B,1] confidence from head (torch tensor)
          findings: textual summary and simple labels (dict)
        """
        # Tokenize - do on CPU to reduce GPU memory pressure
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Forward pass (no grad)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden]
        # Use the token at position 0 (start) or mean pooling (safer)
        note_feat = torch.mean(last_hidden, dim=1)  # [B, hidden]
        note_feat = note_feat.to(self.head[0].weight.device)
        Cm = self.head(note_feat)  # [B,1]
        # crude textual findings: use the model to generate a short summary (lightweight)
        # Generate only if the model is small or generation is acceptable; here we skip heavy generation to conserve resources
        findings = {'note_summary_tokens': int(torch.sum((note_feat.mean(dim=1)>0).long()).item())}
        return note_feat, Cm, findings
