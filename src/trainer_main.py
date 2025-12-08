# src/trainer_main.py
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from models import ImageEncoder, TabularEncoder, NotesEncoder, Aggregator, HeadClassifier, LRM
from dataset import MIMICMultiModalDataset
from trainer import train_epoch, evaluate, DCSModule
from utils import load_yaml, ensure_dir, save_checkpoint, get_device, now_str
import os
from utils import save_json
from metrics import compute_classification_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data_csv', required=True)
    p.add_argument('--img_root', required=True)
    p.add_argument('--ckpt_dir', default='experiments/checkpoints')
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    batch_size = args.batch_size or cfg.get('batch_size', 16)
    device = get_device(prefer_gpu=True)
    print(f"[+] Device: {device}")

    # Dataset
    ds = MIMICMultiModalDataset(args.data_csv, args.img_root)
    n = len(ds)
    val_len = max( int(0.1*n), 1 )
    train_len = n - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # Models
    image_model_name = cfg.get('image_model','swin_base_patch4_window7_224')
    image_out_dim = cfg.get('image_out_dim', 512)
    note_dim = cfg.get('note_dim', 384)
    tab_cols = cfg.get('tabular_cols', ['age','heart_rate','systolic_bp'])
    tab_in = len(tab_cols)

    image_enc = ImageEncoder(model_name=image_model_name, out_dim=image_out_dim, pretrained=True).to(device)
    tab_enc = TabularEncoder(in_dim=tab_in, out_dim=128).to(device)
    notes_enc = NotesEncoder(model_name=cfg.get('notes_model','all-MiniLM-L6-v2'), device=str(device))
    aggregator = Aggregator(im_dim=image_out_dim, tab_dim=128, note_dim=note_dim, hidden=512).to(device)
    head = HeadClassifier(in_dim=512//2, n_classes=2).to(device)  # aggregator final dim = hidden//2
    lrm = LRM(in_dim=512//2).to(device)

    models = (image_enc, tab_enc, aggregator, head, lrm)

    # Optimizers
    params = list(image_enc.parameters()) + list(tab_enc.parameters()) + list(aggregator.parameters()) + list(head.parameters()) + list(lrm.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.get('lr',1e-4))
    optimizers = [optimizer]

    # DCS
    dcs_cfg = cfg.get('dcs', {})
    dcs = DCSModule(w1=dcs_cfg.get('w1',0.4), w2=dcs_cfg.get('w2',0.4), w3=dcs_cfg.get('w3',0.2))

    # training loop
    epochs = cfg.get('epochs', 10)
    ensure_dir(args.ckpt_dir)
    history = {'train_loss':[], 'val_auc_cm':[], 'val_auc_rf':[]}

    for epoch in range(1, epochs+1):
        print(f"=== Epoch {epoch}/{epochs} ===")
        tr_loss = train_epoch(device, models, optimizers, train_loader, notes_enc, dcs, tau=cfg.get('tau',0.6))
        eval_res = evaluate(device, models, val_loader, notes_enc, dcs, tau=cfg.get('tau',0.6))
        print(f"Train loss: {tr_loss:.4f} | val auc cm: {eval_res['auc_cm']:.4f} | val auc rf: {eval_res['auc_rf']:.4f}")

        history['train_loss'].append(tr_loss)
        history['val_auc_cm'].append(eval_res['auc_cm'])
        history['val_auc_rf'].append(eval_res['auc_rf'])

        ckpt_name = f"ckpt_epoch{epoch}_{now_str()}.pt"
        ckpt_path = save_checkpoint({
            'epoch': epoch,
            'state_dicts': {
                'image_enc': image_enc.state_dict(),
                'tab_enc': tab_enc.state_dict(),
                'aggregator': aggregator.state_dict(),
                'head': head.state_dict(),
                'lrm': lrm.state_dict()
            },
            'optimizer': optimizer.state_dict(),
            'cfg': cfg
        }, args.ckpt_dir, name=ckpt_name)
        print(f"[+] Saved checkpoint: {ckpt_path}")

    # save history
    history_path = os.path.join(args.ckpt_dir, 'history.json')
    save_json(history, history_path)
    print("[+] Training finished.")

if __name__ == '__main__':
    main()
