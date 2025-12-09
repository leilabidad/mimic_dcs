"""
Inference pipeline that runs the full multi-agent stack:
- agent_manager.run_agents()
- dcs.combine()
- creates a prediction JSON and writes explain artifacts
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.dataset import MIMICDataset
from src.models import VisionAgent, LabAgent, NotesAgent
from src.lrm import FusionAgent, AgentManager
from src.dcs import DCSAgent
from src.utils import save_json, create_prediction_json

cfg = yaml.safe_load(open("configs/default.yaml"))
device = torch.device(cfg['training']['device'])

dataset = MIMICDataset(cfg['dataset']['clinical_csv'], cfg['dataset']['image_dir'],
                       image_size=cfg['dataset']['image_size'],
                       tabular_cols=cfg['dataset'].get('tabular_cols', []))
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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

out_dir = cfg['paths']['explain_out']
os.makedirs(out_dir, exist_ok=True)

# Optional checkpoint loading if you want to use trained weights
ckpt_dir = cfg['paths']['checkpoints']
ckpt_paths = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)] ) if os.path.exists(ckpt_dir) else []
if ckpt_paths:
    latest = ckpt_paths[-1]
    ckpt = torch.load(latest, map_location=device)
    vision.load_state_dict(ckpt['vision'])
    lab.load_state_dict(ckpt['lab'])
    fusion.load_state_dict(ckpt['fusion'])
    print(f"Loaded checkpoint: {latest}")

vision.eval()
lab.eval()
for batch in loader:
    image = batch['image']
    tab = batch['tabular']
    note = batch['note_text']
    pid = batch['patient_id'][0]

    with torch.no_grad():
        agent_pkg = agent_manager.run_agents(image, tab, note)
        Cm_img = agent_pkg['vision']['Cm']
        Cm_lab = agent_pkg['lab']['Cm']
        Cm_note = agent_pkg['note']['Cm']
        joint = agent_pkg['fusion']['joint']

        dcs_out = dcs.combine(Cm_img.detach().cpu().numpy(), Cm_lab.detach().cpu().numpy(),
                              Cm_note.detach().cpu().numpy(), joint.detach().cpu().numpy())
        Rf = dcs_out['Rf'][0][0]
        qc_flag = dcs_out['explain']['qc_flag'][0]
        final_label = "High Risk" if Rf >= cfg['model']['qc_threshold'] else "Low Risk"

        pred_json = create_prediction_json(pid, agent_pkg, dcs_out['explain'], final_label)
        out_path = os.path.join(out_dir, f"{pid}_prediction.json")
        save_json(pred_json, out_path)
        print("Saved prediction:", out_path)
