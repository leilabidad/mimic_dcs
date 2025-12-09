"""
Local Reasoning Module (LRM) updated to accept joint embedding from aggregator.
It computes Sc (consistency score) and returns an explain dict with top factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalReasoningModule(nn.Module):
    def __init__(self, proj_dim=512, hidden=256, consistency_weight=0.8):
        """
        proj_dim: dimension of joint embedding from aggregator
        hidden: hidden size of MLP
        """
        super().__init__()
        self.consistency_weight = consistency_weight
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        # small explainer head that outputs reasoning tokens (scores) per modality
        self.explain_head = nn.Sequential(
            nn.Linear(proj_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, joint_embedding, attention_maps=None, importance=None):
        """
        joint_embedding: [B, proj_dim]
        attention_maps: dict returned by aggregator (optional) for explainability
        importance: [B,3] modality importance scores (optional)
        Returns:
          Sc: [B,1] consistency score in [0,1]
          explain: dict containing attention + importance + top modality reason
        """
        Sc = self.mlp(joint_embedding)  # [B,1]

        # prepare explainability information
        explain = {}
        if importance is not None:
            explain['importance'] = importance.detach().cpu().tolist()  # per-sample modality importance
        if attention_maps is not None:
            # convert small attention tensors to lists
            explain_attn = {}
            for k, v in attention_maps.items():
                # v shape [B, q_len, k_len] - we convert to CPU list for each batch
                explain_attn[k] = v.detach().cpu().squeeze().tolist()
            explain['attention_maps'] = explain_attn

        # top modality reason (which modality contributed most)
        if importance is not None:
            top_idx = importance.argmax(dim=-1)  # [B]
            modality_names = ['image', 'tabular', 'notes']
            explain['top_modality'] = [modality_names[i] for i in top_idx.detach().cpu().tolist()]

        return Sc, explain
