"""
Trainer script that trains ImageAgent and TabularAgent, uses NotesAgent (LLM) in frozen mode,
and uses DCSSystem (aggregator + LRM) as post-hoc for computing Rf.

Training objective:
 - supervise agents (image/tabular) with BCE on their own Cm outputs (aux losses)
 - supervise final Rf against label (main loss)
 - NotesAgent (LLM) is frozen for efficiency (acts as reasoning agent)
"""

import torch
from torch.utils.data import DataLoader
import yaml
import os

from src.dataset import MIMICDataset
from src.models import ImageAgent, TabularAgent, NotesAgent
from src.dcs import DCSSystem

cfg = yaml.safe_load(open("configs/default.yaml"))
device = torch.device(cfg['training']['device'])

# Data
dataset = MIMICDataset(cfg['dataset']['clinical_csv'], cfg['dataset']['image_dir'], cfg['dataset']['image_size'])
loader = DataLoader(dataset, batch_size=cfg['dataset']['batch_size'], shuffle=True, num_workers=4)

# Agents and DCS system
img_agent = ImageAgent().to(device)
tab_agent = TabularAgent(input_dim=dataset[0]['tabular'].shape[0], hidden_dim=cfg['model']['tabular_hidden']).to(device)
notes_agent = NotesAgent(cfg['model']['notes_model'])  # keep on device_map auto; we will not update its weights
dcs = DCSSystem(img_dim=1024, tab_dim=128, note_dim=4096,
                proj_dim=512, w1=cfg['model']['dcs_weights']['w1'],
                w2=cfg['model']['dcs_weights']['w2'], w3=cfg['model']['dcs_weights']['w3'],
                qc_threshold=cfg['model']['qc_threshold'], device=device).to(device)

# Freeze notes agent parameters (LLM)
for p in notes_agent.parameters():
    p.requires_grad = False

# Optimizer: train image and tabular agent parameters only
opt_params = list(img_agent.backbone.parameters()) + list(img_agent.classifier.parameters()) + list(tab_agent.mlp.parameters()) + list(dcs.parameters())
optimizer = torch.optim.AdamW(opt_params, lr=cfg['training']['lr'])
scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['mixed_precision'])

# Loss functions
bce = torch.nn.BCELoss()

save_dir = "experiments/checkpoints"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(cfg['training']['epochs']):
    img_agent.train()
    tab_agent.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        image = batch['image'].to(device)
        tabular = batch['tabular'].to(device)
        note_texts = batch['note_text']
        label = batch['label'].to(device).unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg['training']['mixed_precision']):
            # Agent forward
            img_feat, Cm_img = img_agent(image)           # Cm_img: [B,1]
            tab_feat, Cm_tab = tab_agent(tabular)        # Cm_tab: [B,1]
            note_feat, Cm_note = notes_agent(note_texts, device)  # Cm_note: [B,1] (note: nodes may be on different device_map)

            # DCS forward (aggregator + LRM + final Rf)
            Rf, qc_flag, explain = dcs(img_feat, tab_feat, note_feat, Cm_img, Cm_tab, Cm_note)

            # Losses:
            # 1) agent-level auxiliary losses (encourage agents to make good predictions)
            loss_img = bce(Cm_img, label)
            loss_tab = bce(Cm_tab, label)
            # 2) final Rf loss (main objective)
            loss_rf = bce(Rf, label)
            # Weighted sum
            loss = 0.5 * loss_rf + 0.25 * loss_img + 0.25 * loss_tab

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

    # Save checkpoint (lightweight: save image/tabular/dcs weights)
    ckpt = {
        'epoch': epoch,
        'img_agent': img_agent.state_dict(),
        'tab_agent': tab_agent.state_dict(),
        'dcs': dcs.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch_{epoch}.pt"))
