"""
DCS agent implemented as pure Python + NumPy functions.
This file intentionally avoids torch so DCS remains a post-hoc, framework-agnostic module.
It consumes agent confidences and a joint_embedding (if available) and produces:
 - Rf: reliability score
 - QC flag: boolean decision
 - explanation: structured diagnostic info
"""

import numpy as np
from typing import Dict, Any, List, Optional

def _safe_to_numpy(x):
    """Convert various tensor/array-like inputs to 1D numpy float array"""
    if x is None:
        return np.array([])
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy().reshape(-1)
        except Exception:
            pass
    if hasattr(x, "numpy"):
        return x.numpy().reshape(-1)
    return np.array(x).reshape(-1)

class DCSAgent:
    """Dynamic Consistency Scoring Agent (post-hoc)"""
    def __init__(self, w_img=0.4, w_lab=0.3, w_note=0.3, qc_threshold=0.75, multiplicative=0.1):
        self.w_img = w_img
        self.w_lab = w_lab
        self.w_note = w_note
        self.qc_threshold = qc_threshold
        self.multiplicative = multiplicative

    def compute_sc_from_joint(self, joint_embedding: Optional[np.ndarray]) -> np.ndarray:
        """Lightweight consistency proxy from joint embedding: normalized variance -> lower variance => higher consistency"""
        if joint_embedding is None or joint_embedding.size == 0:
            return np.array([0.5])
        # compute inverse normalized std as consistency: higher -> more consistent
        std = np.std(joint_embedding, axis=1)
        # normalize to [0,1] by dividing with (max_std + eps)
        max_std = np.max(std) + 1e-6
        sc = 1.0 - (std / max_std)
        return sc.reshape(-1, 1)  # shape [B,1]

    def combine(self, Cm_img, Cm_lab, Cm_note, joint_embedding: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Inputs:
          Cm_*: array-like or torch-like shape [B,1] or [B]
          joint_embedding: optional numpy array [B, dim]
        Returns:
          dict with keys: Rf (numpy [B,1]), qc_flag (list[bool]), explain (dict)
        """
        c_img = _safe_to_numpy(Cm_img)
        c_lab = _safe_to_numpy(Cm_lab)
        c_note = _safe_to_numpy(Cm_note)
        # Ensure shapes
        if c_img.ndim == 0:
            c_img = np.array([c_img])
        if c_lab.ndim == 0:
            c_lab = np.array([c_lab])
        if c_note.ndim == 0:
            c_note = np.array([c_note])
        # reshape to (B,)
        c_img = c_img.reshape(-1)
        c_lab = c_lab.reshape(-1)
        c_note = c_note.reshape(-1)
        B = max(len(c_img), len(c_lab), len(c_note))
        # broadcast smaller arrays
        def _broadcast(a):
            if len(a) == 1 and B > 1:
                return np.full((B,), a[0])
            return a
        c_img = _broadcast(c_img)
        c_lab = _broadcast(c_lab)
        c_note = _broadcast(c_note)

        # compute Sc from joint embedding
        Sc = self.compute_sc_from_joint(joint_embedding)  # [B,1]
        Sc = Sc.reshape(-1)

        # final Rf combining weights + multiplicative interaction
        Rf = (self.w_img * c_img) + (self.w_lab * c_lab) + (self.w_note * c_note) + \
             self.multiplicative * (c_img * c_lab * c_note) + 0.05 * (c_img * Sc)

        # clip to [0,1]
        Rf = np.clip(Rf, 0.0, 1.0)

        qc_flag = (Rf < self.qc_threshold).tolist()
        explain = {
            'Cm_img': c_img.tolist(),
            'Cm_lab': c_lab.tolist(),
            'Cm_note': c_note.tolist(),
            'Sc': Sc.tolist(),
            'Rf': Rf.tolist(),
            'qc_flag': qc_flag
        }
        return {'Rf': Rf.reshape(-1, 1), 'qc_flag': qc_flag, 'explain': explain}
