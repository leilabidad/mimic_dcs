import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ImageEncoder, TabularEncoder, NotesEncoder, Aggregator, HeadClassifier, LRM
from dataset import MIMICMultiModalDataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

class DCSModule:
    def __init__(self, w1=0.4, w2=0.4, w3=0.2):
        self.w1 = w1; self.w2 = w2; self.w3 = w3
    def compute(self, cm, sc):
        # cm: confidence [0,1], sc: [0,1]
        return self.w1*cm + self.w2*sc + self.w3*(cm*sc)

def train_epoch(device, models, optimizers, loader, notes_encoder, dcs_module, tau=0.6):
    image_enc, tab_enc, aggregator, head_cls, lrm = models
    image_enc.train(); tab_enc.train(); aggregator.train(); head_cls.train(); lrm.train()

    losses = []
    for batch in loader:
        imgs = batch['image'].to(device)
        tabs = batch['tabular'].to(device)
        texts = batch['text']
        labels = batch['label'].to(device)

        img_feat = image_enc(imgs)
        tab_feat = tab_enc(tabs)
        note_feat = notes_encoder.encode(texts)  # tensor on device
        if note_feat.device != device: note_feat = note_feat.to(device)

        evidence = aggregator(img_feat, tab_feat, note_feat)
        logits = head_cls(evidence)
        probs = F.softmax(logits, dim=1)[:,1]  # positive class prob = Cm
        cm = probs.detach()   # baseline confidence

        sc = lrm(evidence)    # consistency score in [0,1]

        rf = dcs_module.compute(cm, sc)   # final composed score (tensor)

        # Loss: standard CE on backbone + optional regularizer to align rf with label
        ce = F.cross_entropy(logits, labels)
        # we can add a small loss to push rf toward 1 for positive class, 0 for negative
        rf_target = labels.float()
        rf_loss = F.mse_loss(rf, rf_target)

        loss = ce + 0.5*rf_loss

        # backward
        for opt in optimizers: opt.zero_grad()
        loss.backward()
        for opt in optimizers: opt.step()

        losses.append(loss.item())
    return np.mean(losses)

def evaluate(device, models, loader, notes_encoder, dcs_module, tau=0.6):
    image_enc, tab_enc, aggregator, head_cls, lrm = models
    image_enc.eval(); tab_enc.eval(); aggregator.eval(); head_cls.eval(); lrm.eval()

    all_labels=[]; all_cm=[]; all_rf=[]
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            tabs = batch['tabular'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)

            img_feat = image_enc(imgs)
            tab_feat = tab_enc(tabs)
            note_feat = notes_encoder.encode(texts)
            if note_feat.device != device: note_feat = note_feat.to(device)

            evidence = aggregator(img_feat, tab_feat, note_feat)
            logits = head_cls(evidence)
            probs = F.softmax(logits, dim=1)[:,1]
            cm = probs
            sc = lrm(evidence)
            rf = dcs_module.compute(cm, sc)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_cm.extend(cm.cpu().numpy().tolist())
            all_rf.extend(rf.cpu().numpy().tolist())

    auc_cm = roc_auc_score(all_labels, all_cm)
    auc_rf = roc_auc_score(all_labels, all_rf)
    return {'auc_cm': auc_cm, 'auc_rf': auc_rf}
