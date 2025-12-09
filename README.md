# mimic_dcs


```

mimic_dcs/
├─ data/
│   ├─ images/               # Chest X-ray images (.png/.jpg)
│   └─ clinical.csv          # patient_id, age, labs..., note_text, label(s)
├─ configs/
│   └─ default.yaml
├─ src/
│   ├─ dataset.py
│   ├─ models.py
│   ├─ lrm.py
│   ├─ dcs.py
│   ├─ trainer.py
│   ├─ inference.py
│   ├─ utils.py
│   └─ metrics.py
├─ scripts/
│   ├─ preprocess_mimic.sh
│   └─ run_train.sh
├─ experiments/
└─ README.md



```
