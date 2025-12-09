"""
Cross-modal Agent Aggregator with multi-head attention.
This module takes per-agent features and produces:
 - joint_embedding: fused representation for downstream reasoning
 - attention_maps: dict of attention matrices for explainability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAggregator(nn.Module):
    def __init__(self, img_dim=1024, tab_dim=128, note_dim=4096, proj_dim=512, n_heads=8, dropout=0.1):
        """
        img_dim, tab_dim, note_dim: input feature dims from each agent
        proj_dim: common projection dimension for attention
        n_heads: number of attention heads
        """
        super().__init__()
        self.proj_dim = proj_dim
        # linear projections to common dimension
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.tab_proj = nn.Linear(tab_dim, proj_dim)
        self.note_proj = nn.Linear(note_dim, proj_dim)
        # multi-head attention layers for pairwise attention
        self.attn_img_tab = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_img_note = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_tab_note = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        # fusion MLP to produce joint embedding
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        # small layer to produce modality importance scores for explainability
        self.importance_head = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # importance for [img, tab, note]
            nn.Softmax(dim=-1)
        )

    def forward(self, img_feat, tab_feat, note_feat):
        """
        Inputs:
          img_feat: [batch, img_dim]
          tab_feat: [batch, tab_dim]
          note_feat: [batch, note_dim]
        Outputs:
          joint_embedding: [batch, proj_dim]
          attention_maps: dict with keys 'img_tab', 'img_note', 'tab_note' containing attention weights
          importance: [batch,3] modality importance scores
        Note: MultiheadAttention expects sequences [batch, seq_len, embed], we will use seq_len=1 for each modality vector.
        """
        # Project to common dim
        q_img = self.img_proj(img_feat).unsqueeze(1)   # [B,1,proj]
        k_img = q_img
        q_tab = self.tab_proj(tab_feat).unsqueeze(1)
        k_tab = q_tab
        q_note = self.note_proj(note_feat).unsqueeze(1)
        k_note = q_note

        # Pairwise attention: query=first, key/value=second (we compute both directions by swapping args if needed)
        # img <- tab
        attn_out_img_tab, attn_w_img_tab = self.attn_img_tab(query=q_img, key=k_tab, value=k_tab, need_weights=True)
        # img <- note
        attn_out_img_note, attn_w_img_note = self.attn_img_note(query=q_img, key=k_note, value=k_note, need_weights=True)
        # tab <- note
        attn_out_tab_note, attn_w_tab_note = self.attn_tab_note(query=q_tab, key=k_note, value=k_note, need_weights=True)

        # squeeze seq dim
        attn_out_img_tab = attn_out_img_tab.squeeze(1)   # [B,proj]
        attn_out_img_note = attn_out_img_note.squeeze(1)
        attn_out_tab_note = attn_out_tab_note.squeeze(1)

        # Concatenate fused outputs and pass through fusion MLP
        concat = torch.cat([attn_out_img_tab, attn_out_img_note, attn_out_tab_note], dim=1)  # [B, proj*3]
        joint_embedding = self.fusion(concat)  # [B,proj]

        # importance scores from the joint embedding
        importance = self.importance_head(joint_embedding)  # [B,3] values sum to 1

        # Prepare attention maps for explainability (squeeze head/seq dims)
        # attn_w: [B, num_heads, query_len, key_len] but batch_first=True => returns [B, query_len, key_len] averaged by heads by default
        # PyTorch MultiheadAttention with need_weights=True returns (attn_output, attn_output_weights) where attn_output_weights shape: [B, query_len, key_len]
        attention_maps = {
            'img_tab': attn_w_img_tab,   # [B,1,1] typically
            'img_note': attn_w_img_note,
            'tab_note': attn_w_tab_note
        }

        return joint_embedding, attention_maps, importance
