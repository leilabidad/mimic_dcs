import torch
import json

def infer_single(sample, models, notes_encoder, dcs_module, device='cpu', tau=0.6):
    image_enc, tab_enc, aggregator, head_cls, lrm = models
    image_enc.eval(); tab_enc.eval(); aggregator.eval(); head_cls.eval(); lrm.eval()

    img = sample['image'].unsqueeze(0).to(device)
    tab = sample['tabular'].unsqueeze(0).to(device)
    text = [sample['text']]

    img_feat = image_enc(img)
    tab_feat = tab_enc(tab)
    note_feat = notes_encoder.encode(text).to(device)
    evidence = aggregator(img_feat, tab_feat, note_feat)
    logits = head_cls(evidence)
    probs = torch.softmax(logits, dim=1)[0,1].item()
    cm = probs
    sc = lrm(evidence).item()
    rf = dcs_module.compute(torch.tensor(cm), torch.tensor(sc)).item()
    qc_flag = rf < tau

    out = {
        'pred_prob': cm,
        'Sc': sc,
        'Rf': rf,
        'QC_flag': bool(qc_flag),
        'pred_label': int(cm >= 0.5)
    }
    return out
