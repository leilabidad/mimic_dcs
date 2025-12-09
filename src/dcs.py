"""
DCS module integrates:
 - per-agent confidences (Cm_img, Cm_tab, Cm_note)
 - aggregator joint_embedding, attention maps, importance
 - LRM outputs (Sc + explain)
Then computes final Rf and QC flag.
"""

import torch
import torch.nn as nn
from src.lrm import LocalReasoningModule
from src.aggregator import CrossModalAggregator

class DCSSystem(nn.Module):
    def __init__(self, img_dim=1024, tab_dim=128, note_dim=4096, proj_dim=512, 
                 w1=0.4, w2=0.3, w3=0.3, qc_threshold=0.75, device='cuda'):
        super().__init__()
        self.device = device
        self.aggregator = CrossModalAggregator(img_dim=img_dim, tab_dim=tab_dim, note_dim=note_dim, proj_dim=proj_dim).to(device)
        self.lrm = LocalReasoningModule(proj_dim=proj_dim).to(device)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.qc_threshold = qc_threshold

    def forward(self, img_feat, tab_feat, note_feat, Cm_img, Cm_tab, Cm_note):
        """
        Inputs:
          img_feat, tab_feat, note_feat: per-agent features
          Cm_img, Cm_tab, Cm_note: per-agent confidences (all tensors shaped [B,1])
        Outputs:
          Rf: final reliability score [B,1]
          qc_flag: boolean tensor [B,1]
          explain: dict with LRM explain outputs + agent confidences + aggregator importance/attn
        """
        # Aggregate cross-modal information
        joint_embedding, attention_maps, importance = self.aggregator(img_feat, tab_feat, note_feat)
        # LRM computes consistency Sc and explanation
        Sc, explain_lrm = self.lrm(joint_embedding, attention_maps=attention_maps, importance=importance)
        # Combine confidences (weighted + multiplicative interactions)
        # Ensure Cm_* and Sc are same dtype/device
        Cm_img = Cm_img.to(self.device)
        Cm_tab = Cm_tab.to(self.device)
        Cm_note = Cm_note.to(self.device)
        Sc = Sc.to(self.device)
        Rf = self.w1 * Cm_img + self.w2 * Cm_tab + self.w3 * Cm_note + 0.1 * (Cm_img * Cm_tab * Cm_note) + 0.05 * (Cm_img * Sc)
        qc_flag = (Rf < self.qc_threshold)
        # Build explain dict
        explain = {
            'agent_confidences': {
                'Cm_img': Cm_img.detach().cpu().tolist(),
                'Cm_tab': Cm_tab.detach().cpu().tolist(),
                'Cm_note': Cm_note.detach().cpu().tolist(),
            },
            'Sc': Sc.detach().cpu().tolist(),
            'Rf': Rf.detach().cpu().tolist(),
            'qc_flag': qc_flag.detach().cpu().tolist(),
            'aggregator_importance': importance.detach().cpu().tolist(),
        }
        # merge LRM explainables
        explain.update({'lrm_explain': explain_lrm})
        return Rf, qc_flag, explain
