# src/utils.py
import os
import json
import torch
import yaml
from datetime import datetime

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_checkpoint(state, ckpt_dir, name='last.pt'):
    ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, name)
    torch.save(state, path)
    return path

def load_checkpoint(path, device='cpu'):
    if not path or not os.path.exists(path):
        return None
    return torch.load(path, map_location=device)

def now_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
