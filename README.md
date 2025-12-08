# mimic_dcs


```
mimic_dcs/
├─ data/
│  ├─ images/               # chest xray .png/.jpg (paths stored in csv)
│  └─ clinical.csv          # patient_id, age, labs..., note_text, label(s)
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ dataset.py
│  ├─ models.py
│  ├─ trainer.py
│  ├─ inference.py
│  ├─ metrics.py
│  └─ utils.py
├─ scripts/
│  ├─ preprocess_mimic.sh
│  └─ run_train.sh
├─ experiments/
└─ README.md


```
