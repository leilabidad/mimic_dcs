from graphviz import Source

dot_pre_setup = r"""
digraph MIMIC_PreSetup_DCS {
    fontname=Helvetica fontsize=12 rankdir=TB size="14,20" splines=ortho nodesep=0.6 ranksep=0.9

    // ENVIRONMENT
    Env [label="Check Environment\n- Python >= 3.10\n- GPU available\n- CUDA/cuDNN installed" fillcolor="#A6CEE3" style=filled shape=box]

    // DEPENDENCIES
    InstallDeps [label="Install Python Packages\n- torch, torchvision, timm\n- transformers, pandas, numpy\n- graphviz" fillcolor="#B2DF8A" style=filled shape=box]

    // DATASET
    PrepareDataset [label="Download & Preprocess MIMIC Dataset\n- Chest X-rays\n- Clinical Notes\n- Labs/Vitals\n- Train/Val/Test split" fillcolor="#FB9A99" style=filled shape=box]

    // BASE MODELS
    DownloadSwin [label="Download/Prepare Swin Transformer\n- Pretrained on CXR\n- Fine-tune on MIMIC if needed" fillcolor="#FFD966" style=filled shape=component]
    DownloadEncoders [label="Prepare Encoders\n- Tabular Encoder (MLP/AutoEncoder)\n- Clinical Notes Encoder (BioClinicalBERT)" fillcolor="#D5D8DC" style=filled shape=component]

    // TRAIN BASE MODELS
    TrainBase [label="Train/Fine-tune Base Models\n- SwinNet\n- TabEncoder\n- NoteEncoder" fillcolor="#FDBF6F" style=filled shape=box]

    // DCS WEIGHTS
    TrainDCS [label="Determine DCS Weights (w1, w2, w3)\n- Optimize on validation set\n- Save for inference" fillcolor="#FF7F00" style=filled shape=box]

    // SAVE ARTIFACTS
    SaveArtifacts [label="Save All Models & Preprocessed Data\n- Checkpoints & embeddings\n- DCS weights\n- Preprocessed JSON" fillcolor="#CAB2D6" style=filled shape=box]

    // FLOW
    Env -> InstallDeps -> PrepareDataset
    PrepareDataset -> DownloadSwin
    PrepareDataset -> DownloadEncoders
    DownloadSwin -> TrainBase
    DownloadEncoders -> TrainBase
    TrainBase -> TrainDCS -> SaveArtifacts
}
"""

s = Source(dot_pre_setup)
s.render('MIMIC_PreSetup_DCS_Flowchart', format='png', cleanup=True)
print("Done! Generated: MIMIC_PreSetup_DCS_Flowchart.png")
