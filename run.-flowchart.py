from graphviz import Source

dot_source = r"""
digraph MIMIC_Agentic_New_Architecture {
    fontname=Helvetica fontsize=12 rankdir=TB size="14,26" splines=ortho nodesep=0.6 ranksep=0.9

    // INPUTS
    X [label="Chest X-ray\n{'image': 'path/to/img.png'}" fillcolor="#A6CEE3" style=filled shape=box]
    T [label="Tabular Labs/Vitals\n{'labs':..., 'vitals':..., 'age': 65}" fillcolor="#B2DF8A" style=filled shape=box]
    N [label="Clinical Notes\n{'clinical_notes': '...'}" fillcolor="#FB9A99" style=filled shape=box]

    // BASE MODELS / FEATURE EXTRACTORS
    Swin2D [label="Swin Transformer 2D\n(Pretrained on CXR datasets)\n→ img_embedding" fillcolor="#FFD966" style=filled shape=component]
    TabEncoder [label="Tabular Encoder\n(MLP / AutoEncoder)\n→ lab_embedding" fillcolor="#D5D8DC" style=filled shape=component]
    NoteEncoder [label="Clinical Notes Encoder\n(BioClinicalBERT)\n→ note_embedding" fillcolor="#D5D8DC" style=filled shape=component]

    // AGENTS
    VisionAgent [label="Agent 1: Vision\nConsumes img_embedding\n→ {'img_feat':..., 'Cm':0.78}" fillcolor="#FDBF6F" style=filled shape=ellipse]
    LabAgent [label="Agent 2: Lab\nConsumes lab_embedding\n→ {'lab_feat':..., 'risk_score':0.65}" fillcolor="#FDBF6F" style=filled shape=ellipse]
    NoteAgent [label="Agent 3: Notes\nConsumes note_embedding\n→ {'note_feat':..., 'flags':['possible_pneumonia']}" fillcolor="#FDBF6F" style=filled shape=ellipse]

    FusionAgent [label="Agent 4: Fusion (LRM)\n→ {'evidence_bundle':..., 'Sc':0.85}" fillcolor="#CAB2D6" style=filled shape=diamond]
    DCSAgent [label="Agent 5: DCS\nRf = w1*Cm + w2*Sc + w3*(Cm*Sc)\n→ {'final_conf':0.80}" fillcolor="#FF7F00" style=filled shape=folder]

    // QC & DECISION
    DM [label="Decision Manager\n→ {'final_label':'High Risk','Rf_score':0.80}" fillcolor="#FFE599" style=filled shape=box]
    QCnode [label="QC Enabled?" shape=diamond style=filled fillcolor="#EAD1DC"]
    QCcheck [label="Rf ≥ τ ?" shape=diamond style=filled fillcolor="#B6D7A8"]
    Manual [label="Manual Review\n(Loop → Fusion/DCS)" fillcolor="#F9CB9C" style=filled shape=box]
    Accept [label="Accept Prediction" fillcolor="#B6D7A8" style=filled shape=box]

    // OUTCOME
    OUTCOME [label="MIMIC-III/IV Ground Truth\n{'in_hospital_mortality':1,...}" fillcolor="#FFCCCC" style=filled shape=box]
    EVAL [label="Evaluation Module\nAUROC, F1, Calibration" fillcolor="#CCE5FF" style=filled shape=box]

    // OUTPUT JSON
    FINAL [label="Final Output JSON\n{'patient_id':12345,'pred_label':'High Risk','Cm':0.78,'Sc':0.85,'Rf':0.80,'QC_flag':True,'issues':['vitals anomaly']}" fillcolor="#FFFFFF" style=filled shape=box]

    // FLOW

    // Inputs -> Base models
    X -> Swin2D
    T -> TabEncoder
    N -> NoteEncoder

    // Base models -> Agents
    Swin2D -> VisionAgent
    TabEncoder -> LabAgent
    NoteEncoder -> NoteAgent

    // Agents -> Fusion
    VisionAgent -> FusionAgent
    LabAgent -> FusionAgent
    NoteAgent -> FusionAgent

    // Fusion -> DCS -> Decision
    FusionAgent -> DCSAgent
    DCSAgent -> DM

    // Decision -> QC
    DM -> QCnode
    QCnode:sw -> QCcheck:nw [label="Yes" color=blue]
    QCnode:se -> FINAL:n [label="No QC" style=dashed color=gray]

    QCcheck:sw -> Manual:nw [label="Rf < τ" color=red]
    QCcheck:se -> Accept:nw [label="Rf ≥ τ" color=green]

    Manual -> FusionAgent [style=dashed color=blue]
    Accept -> FINAL

    // Outcome Evaluation
    FINAL -> EVAL
    OUTCOME -> EVAL [color=red penwidth=2]
}
"""

s = Source(dot_source)
s.render('MIMIC_Agentic_New_Flowchart', format='png', cleanup=True)
print("Done! Generated: MIMIC_Agentic_New_Flowchart.png")
