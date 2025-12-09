"""
Trainer script:
- trains VisionAgent and LabAgent (supervised)
- NotesAgent is used as frozen reasoning agent (unless you want to fine-tune)
- DCSAgent is post-hoc and not trained
- FusionAgent is trained jointly with vision/lab when possible
"""

import os
import yaml
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MIMICDataset
from src.models import VisionAgent, LabAgent, NotesAgent
from src.lrm import FusionAgent, AgentManager
from src.dcs import DCSAgent

# Load config
cfg = yaml.safe_load(open("configs/default.yaml"))
device = torch.device(cfg['training']['device'])

# Seed
seed = cfg.get('misc', {}).get('seed', 42)
random.seed(seed)
torch.manual_seed(seed)

# Dataset
dataset = MIMICDataset(cfg['dataset']['clinical_csv'], cfg['dataset']['image_dir'],
                       image_size=cfg['dataset']['image_size'],
                       tabular_cols=cfg['dataset'].get('tabular_cols', []))
loader = DataLoader(dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True,
                    num_workers=cfg.get('misc', {}).get('num_workers', 4))

# Build agents
vision = VisionAgent(backbone_name=cfg['model']['image_backbone']).to(device)
lab = LabAgent(input_dim=len(dataset.tab_cols), hidden=cfg['model']['tabular_hidden']).to(device)
notes = NotesAgent(model_name=cfg['model']['notes_model'], device=device)
fusion = FusionAgent(img_dim=1024, lab_dim=lab.mlp[-1].out_features if hasattr(lab.mlp[-1], 'out_features') else 64,
                     note_dim=notes.model.config.hidden_size, proj_dim=512).to(device)

agent_manager = AgentManager(vision, lab, notes, fusion, device=device)
dcs = DCSAgent(w_img=cfg['model']['dcs_weights']['w_img'],
               w_lab=cfg['model']['dcs_weights']['w_lab'],
               w_note=cfg['model']['dcs_weights']['w_note'],
               qc_threshold=cfg['model']['qc_threshold'])

# Freeze notes model if requested
if cfg['model'].get('freeze_notes_model', True):
    # Notes agent is not a torch.nn.Module for weights, but the underlying model is; request no grads
    for p in notes.model.parameters():
        p.requires_grad = False

# Optimizer: vision, lab, fusion parameters + fusion head
opt_params = list(vision.parameters()) + list(lab.parameters()) + list(fusion.parameters())
optimizer = torch.optim.AdamW(opt_params, lr=cfg['training']['lr'])
scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['mixed_precision'])
bce = torch.nn.BCELoss()

# Checkpoint dir
ckpt_dir = cfg['paths']['checkpoints']
os.makedirs(ckpt_dir, exist_ok=True)

# Training loop
for epoch in range(cfg['training']['epochs']):
    vision.train()
    lab.train()
    fusion.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        tabs = batch['tabular'].to(device)
        notes_texts = batch['note_text']
        labels = batch['label'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg['training']['mixed_precision']):
            # Run agents
            agent_package = agent_manager.run_agents(images, tabs, notes_texts)
            Cm_img = agent_package['vision']['Cm']
            Cm_lab = agent_package['lab']['Cm']
            Cm_note = agent_package['note']['Cm']
            joint = agent_package['fusion']['joint']
            # DCS is not trainable here — get final Rf for supervision
            dcs_out = dcs.combine(Cm_img.detach().cpu().numpy(), Cm_lab.detach().cpu().numpy(),
                                  Cm_note.detach().cpu().numpy(), joint.detach().cpu().numpy())
            Rf = torch.tensor(dcs_out['Rf'], dtype=torch.float32, device=device)  # [B,1]
            # Loss: combine agent-level auxiliary losses + final Rf supervision
            loss_img = bce(Cm_img, labels)
            loss_lab = bce(Cm_lab, labels)
            loss_rf = bce(Rf, labels)
            loss = 0.5 * loss_rf + 0.25 * loss_img + 0.25 * loss_lab

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    ckpt = {
        'epoch': epoch,
        'vision': vision.state_dict(),
        'lab': lab.state_dict(),
        'fusion': fusion.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pt"))
