"""
Inference script that runs the three agents, runs DCSSystem,
and writes explain JSON and optionally attention plots for inspection.
"""

import torch
from torch.utils.data import DataLoader
import yaml
import os

from src.dataset import MIMICDataset
from src.models import ImageAgent, TabularAgent, NotesAgent
from src.dcs import DCSSystem
from src.utils import create_output_json
from src.explain import save_explain_json, plot_attention_map

cfg = yaml.safe_load(open("configs/default.yaml"))
device = torch.device(cfg['training']['device'])

dataset = MIMICDataset(cfg['dataset']['clinical_csv'], cfg['dataset']['image_dir'], cfg['dataset']['image_size'])
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize models (load checkpoints if available)
img_agent = ImageAgent().to(device)
tab_agent = TabularAgent(input_dim=dataset[0]['tabular'].shape[0], hidden_dim=cfg['model']['tabular_hidden']).to(device)
notes_agent = NotesAgent(cfg['model']['notes_model'])
dcs = DCSSystem(img_dim=1024, tab_dim=128, note_dim=4096,
                proj_dim=512, w1=cfg['model']['dcs_weights']['w1'],
                w2=cfg['model']['dcs_weights']['w2'], w3=cfg['model']['dcs_weights']['w3'],
                qc_threshold=cfg['model']['qc_threshold'], device=device).to(device)

# Optionally load a checkpoint (if exists)
ckpt_path = "experiments/checkpoints/ckpt_epoch_last.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    img_agent.load_state_dict(ckpt['img_agent'])
    tab_agent.load_state_dict(ckpt['tab_agent'])
    dcs.load_state_dict(ckpt['dcs'])
    print("Loaded checkpoint:", ckpt_path)

img_agent.eval()
tab_agent.eval()

out_dir = "experiments/explain_outputs"
os.makedirs(out_dir, exist_ok=True)

for batch in loader:
    image = batch['image'].to(device)
    tabular = batch['tabular'].to(device)
    note_texts = batch['note_text']
    patient_id = str(batch['patient_id'][0])

    with torch.no_grad():
        img_feat, Cm_img = img_agent(image)
        tab_feat, Cm_tab = tab_agent(tabular)
        note_feat, Cm_note = notes_agent(note_texts, device)
        Rf, qc_flag, explain = dcs(img_feat, tab_feat, note_feat, Cm_img, Cm_tab, Cm_note)

        final_label = "High Risk" if float(Rf.detach().cpu().item()) >= cfg['model']['qc_threshold'] else "Low Risk"
        out_json = create_output_json(patient_id, float(Cm_img.detach().cpu()), float(Cm_tab.detach().cpu()),
                                      float(Cm_note.detach().cpu()), float(Rf.detach().cpu()), final_label,
                                      Sc=explain.get('Sc', None), issues=[])

        # Save main output
        print(out_json)

        # Save explain JSON
        explain_fname = save_explain_json(out_dir, patient_id, explain)

        # Save small attention plots (if present)
        attn_maps = explain.get('lrm_explain', {}).get('attention_maps', {})
        for k, v in attn_maps.items():
            try:
                arr = v
                if isinstance(arr, list):
                    # convert to numpy
                    import numpy as np
                    mat = np.array(arr)
                else:
                    mat = v
                plot_attention_map(mat, title=f"{patient_id}_{k}", out_file=os.path.join(out_dir, f"{patient_id}_{k}.png"))
            except Exception as e:
                # skip plotting if shapes unexpected
                pass

        # Save explain file path
        print("Explain saved:", explain_fname)
