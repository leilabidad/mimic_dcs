"""
Utilities for saving attention maps and explanation outputs to disk for inspection.
"""

import json
import os
import numpy as np
from matplotlib import pyplot as plt

def save_explain_json(out_path, patient_id, explain_dict):
    """Save the explain dictionary to a JSON file."""
    os.makedirs(out_path, exist_ok=True)
    fname = os.path.join(out_path, f"{patient_id}_explain.json")
    with open(fname, "w") as f:
        json.dump(explain_dict, f, indent=2)
    return fname

def plot_attention_map(attn_matrix, title=None, out_file=None):
    """
    Plot a small attention matrix. attn_matrix expected as 2D numpy array.
    """
    plt.figure(figsize=(3,3))
    plt.imshow(attn_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    if title:
        plt.title(title)
    if out_file:
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
