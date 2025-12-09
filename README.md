# mimic_dcs

Multi-agent clinical reasoning pipeline for MIMIC-CXR using Dynamic Consistency Scoring (DCS). This repository implements a full end-to-end multi-modal system (X-ray + labs + notes) with **5 autonomous agents**, an LRM (Large Reasoning Model) fusion controller, and a deterministic DCS engine for conflict resolution.


## Project Structure

```

mimic_dcs/
├─ data/
│   ├─ images/                # Chest X-ray .png/.jpg
│   ├─ raw_clinical.csv       # Raw CSV before preprocessing
│   └─ clinical.csv           # Cleaned CSV with {patient_id, age, labs…, note_text, label}
├─ configs/
│   └─ default.yaml           # Training + model + agent configuration
├─ src/
│   ├─ dataset.py             # Unified multimodal dataset loader
│   ├─ models.py              # Vision + Tabular + Text encoders
│   ├─ lrm.py                 # Agent manager + 5 autonomous agents
│   ├─ dcs.py                 # Deterministic scoring + consistency engine
│   ├─ trainer.py             # Full training loop
│   ├─ inference.py           # Multi-agent inference pipeline
│   ├─ utils.py               # Logging, transforms, helpers
│   └─ metrics.py             # AUROC, F1, calibration
├─ scripts/
│   ├─ preprocess_mimic.sh    # Build clinical.csv and clean text
│   └─ run_train.sh           # One-command training launcher
├─ experiments/
│   └─ checkpoints/           # Saved models
└─ README.md

````


## Installation

```bash
git clone https://github.com/yourname/mimic_dcs
cd mimic_dcs
pip install -r requirements.txt
````

> If using a small GPU, set the notes model to a smaller LLaMA variant in `configs/default.yaml`.

---

## Data Preparation

### Required files

* `data/images/` → MIMIC-CXR PNG/JPG files
* `data/raw_clinical.csv` → Must contain:

```
patient_id, age, labs_json, note_text, label
```

### Preprocessing

```bash
./scripts/preprocess_mimic.sh
```

This script:

* Cleans clinical notes
* Normalizes lab values
* Merges demographic metadata
* Produces `data/clinical.csv` ready for training

---

## Config File — `configs/default.yaml`

```yaml
vision_encoder: "resnet50"
text_model: "llama-3.1-8b"
tabular_dim: 32

agents:
  vision: true
  lab: true
  note: true
  fusion: true
  dcs: true

train:
  batch_size: 16
  lr: 2e-4
  epochs: 10

paths:
  clinical_csv: "data/clinical.csv"
  image_root: "data/images/"
```


## Multi-Agent System Overview

The system has **5 fully autonomous agents**, each with independent goals and JSON outputs:

### Agent 1 — Vision Agent**: Analyze X-ray

```json
{
  "lungs": "mild_infiltration",
  "heart_size": "borderline_enlarged",
  "risk_score": 0.71
}
```

### Agent 2 — Lab Agent**: Interpret lab values

```json
{
  "lab_anomalies": ["hyponatremia"],
  "sepsis_score": 0.42
}
```

### Agent 3 — Note Agent**: Process clinical notes

```json
{
  "summary": "...",
  "findings": [...],
  "flags": ["possible_pneumonia"]
}
```

### Agent 4 — Fusion Agent (LRM)**: Combines Agents 1–3 into a structured reasoning chain.

### Agent 5 — DCS Agent**: Detect contradictions, score consistency, and produce final output.

---

## Dynamic Consistency Scoring (DCS)

DCS performs:

* Cross-agent contradiction detection
* Confidence alignment
* Outlier suppression
* Weighted late fusion
* Final JSON decision

Example output:

```json
{
  "final_label": "pneumonia",
  "confidence": 0.87,
  "consistency_score": 0.92
}
```


## Training

```bash
./scripts/run_train.sh
```

Or manually:

```bash
python src/trainer.py --config configs/default.yaml
```

Training includes:

* Mixed-precision
* Vision encoder fine-tuning
* Tabular encoder training
* LLM projection head training
* Multi-agent supervised fusion pipeline



## Inference

```bash
python src/inference.py --image path/to/image.png --patient_id 12345
```

Output:

```json
{
  "vision_agent": {...},
  "lab_agent": {...},
  "note_agent": {...},
  "fusion_agent": {...},
  "dcs_agent": {
    "final_label": "pulmonary_edema",
    "confidence": 0.81,
    "consistency_score": 0.88
  }
}
```


## Pipeline Diagram (ASCII, GitHub-ready)

```
       ┌──────────────┐
       │ Vision Agent │
       └───────┬──────┘
               │
┌──────────────┴──────────────┐
│       Fusion Agent (LRM)     │
└──────────────┬──────────────┘
               │
       ┌───────┴───────┐
       │ DCS Agent      │
       └───────┬───────┘
               │
        Final Prediction
               
       ┌──────────────┐
       │ Lab Agent    │
       └──────────────┘
       ┌──────────────┐
       │ Note Agent   │
       └──────────────┘
```


## Metrics

* AUROC
* F1 Score
* Calibration error
* Consistency improvement (Δ consistency after DCS)

---

## Experiments

All checkpoints and experiment logs are saved under:

```
experiments/
├─ checkpoints/
├─ fusion_logs/
└─ dcs_analysis/
```


## License

MIT



## Citation

Placeholder for future paper citation.

```
