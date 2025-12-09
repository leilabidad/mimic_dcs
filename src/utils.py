"""
Utility helpers for IO, JSON outputs, saving explain artifacts.
"""

import json
import os
from typing import Any, Dict

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def create_prediction_json(patient_id, agent_package: Dict, dcs_explain: Dict, final_label: str):
    """Create a consolidated JSON to store and audit the decision."""
    out = {
        'patient_id': patient_id,
        'agents': {
            'vision': agent_package['vision']['findings'],
            'lab': agent_package['lab']['findings'],
            'note': agent_package['note']['findings']
        },
        'dcs': dcs_explain,
        'final_label': final_label
    }
    return out
