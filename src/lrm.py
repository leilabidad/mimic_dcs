"""
Agent Manager and Fusion Agent.
- Orchestrates the 4 first agents:
  VisionAgent, LabAgent, NotesAgent, FusionAgent (LLM-based aggregator)
- FusionAgent uses a cross-modal attention fusion (lightweight) to produce a joint embedding
  which will be passed to DCS for final reliability scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List

# We'll reuse a small cross-modal attention fusion similar to CrossModalAggregator
class FusionAgent(nn.Module):
    """
    FusionAgent performs cross-modal attention among agent embeddings and produces:
      - joint_embedding: fused [B,proj_dim]
      - attention_maps: small explainable attention matrices
    """
    def __init__(self, img_dim=1024, lab_dim=64, note_dim=1024, proj_dim=512, n_heads=8):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.lab_proj = nn.Linear(lab_dim, proj_dim)
        self.note_proj = nn.Linear(note_dim, proj_dim)
        self.attn_img_lab = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads, batch_first=True)
        self.attn_img_note = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim*3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.importance = nn.Sequential(nn.Linear(proj_dim, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1))

    def forward(self, img_feat: torch.Tensor, lab_feat: torch.Tensor, note_feat: torch.Tensor):
        # project to common dimension and add seq dim
        q_img = self.img_proj(img_feat).unsqueeze(1)   # [B,1,proj]
        k_lab = self.lab_proj(lab_feat).unsqueeze(1)
        k_note = self.note_proj(note_feat).unsqueeze(1)
        out_img_lab, w_img_lab = self.attn_img_lab(query=q_img, key=k_lab, value=k_lab, need_weights=True)
        out_img_note, w_img_note = self.attn_img_note(query=q_img, key=k_note, value=k_note, need_weights=True)
        out_img_lab = out_img_lab.squeeze(1)
        out_img_note = out_img_note.squeeze(1)
        out_lab_note = k_lab.squeeze(1)  # fallback if needed
        concat = torch.cat([out_img_lab, out_img_note, out_lab_note], dim=1)
        joint = self.fusion(concat)
        imp = self.importance(joint)
        attention_maps = {'img_lab': w_img_lab, 'img_note': w_img_note}
        return joint, attention_maps, imp

class AgentManager:
    """
    AgentManager orchestrates all agents.
    It calls each agent, collects their outputs and returns a package for DCS to consume.
    """
    def __init__(self, vision_agent, lab_agent, notes_agent, fusion_agent, device='cuda'):
        self.vision_agent = vision_agent
        self.lab_agent = lab_agent
        self.notes_agent = notes_agent
        self.fusion_agent = fusion_agent.to(device)
        self.device = device

    def run_agents(self, image: torch.Tensor, tabular: torch.Tensor, note_texts: List[str]):
        """
        Run all agents end-to-end and return structured outputs for DCS.
        Returns a dict:
          {
            'vision': {'feat', 'Cm', 'findings'},
            'lab':    {'feat', 'Cm', 'findings'},
            'note':   {'feat', 'Cm', 'findings'},
            'fusion': {'joint', 'attention', 'importance'}
          }
        """
        # Ensure inputs are on device
        image = image.to(self.device)
        tabular = tabular.to(self.device)

        # Vision agent
        img_feat, Cm_img, vis_findings = self.vision_agent(image)
        # Lab agent
        lab_feat, Cm_lab, lab_findings = self.lab_agent(tabular)
        # Notes agent (may live across devices internally)
        note_feat, Cm_note, note_findings = self.notes_agent.encode_notes(note_texts)

        # Fuse
        # Ensure note_feat dim matches expected (may be large; project if needed)
        # If note_feat is huge, reduce via a linear projection handled in FusionAgent init
        joint, attention_maps, importance = self.fusion_agent(img_feat, lab_feat, note_feat)

        out = {
            'vision': {'feat': img_feat, 'Cm': Cm_img, 'findings': vis_findings},
            'lab':    {'feat': lab_feat, 'Cm': Cm_lab, 'findings': lab_findings},
            'note':   {'feat': note_feat, 'Cm': Cm_note, 'findings': note_findings},
            'fusion': {'joint': joint, 'attention': attention_maps, 'importance': importance}
        }
        return out
