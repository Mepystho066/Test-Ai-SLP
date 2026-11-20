#!/usr/bin/env python3
"""Hito A — pre-entrenamiento del encoder de pose con señas aisladas (LSC50).

No es un trainer de traducción: LSC50 son 50 señas aisladas, así que aquí se entrena
el **encoder** con una cabeza lineal de clasificación. Métrica: accuracy y F1 macro.
Cuando exista frase continua, ese encoder se carga en el seq2seq del fork
(los nombres de módulo coinciden: `sgn_embed.*`, `encoder.*`).

Datos (manifest TSV, sin cabecera):
    <name>\\t<seq_len>\\t<label>

    python train.py --manifest data/lsc50/train.tsv --feat-dir data/lsc50/features \
        --epochs 30 --batch 16 --classes 50

Sin datos todavía:
    python train.py --synthetic --classes 8 --epochs 15     # prueba el pipeline

Salida: models/classifier/best.pt (+ last.pt), history.json, y el state_dict del
encoder listo para transferir (`encoder_only.pt`).
"""
import argparse
import json
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pose_contract as pc
from signjoey.embeddings import SpatialEmbeddings
from signjoey.encoders import TransformerEncoder


# --------------------------------------------------------------------------- datos
class PoseClfDataset(Dataset):
    """Secuencias (T_enc, DIM) con etiqueta de clase."""

    def __init__(self, manifest: str, feat_dir: str, t_max: int = pc.T_ENC):
        self.items: List[Tuple[str, int, str]] = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    parts = line.split("\t")
                    name, seq_len, label = parts[:3]
                    self.items.append((name, int(seq_len), label))
        if not self.items:
            raise ValueError(f"manifest vacío: {manifest}")
        self.feat_dir = feat_dir
        self.t_max = t_max
        self.labels = sorted({lab for _, _, lab in self.items})
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, _, label = self.items[i]
        x = np.load(os.path.join(self.feat_dir, f"{name}.npy")).astype(np.float32)[: self.t_max]
        if x.shape[1] != pc.DIM:
            raise ValueError(f"{name}: se esperaba DIM={pc.DIM}, llegó {x.shape}")
        return torch.from_numpy(x), self.label_to_idx[label]


def collate_clf(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats])
    t = int(lengths.max())
    sgn = torch.zeros(len(feats), t, feats[0].shape[1])
    for i, f in enumerate(feats):
        sgn[i, : f.shape[0]] = f
    mask = (torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1)
    return sgn, mask, lengths, torch.tensor(labels)


def make_synthetic(root: str, n_per_class: int, classes: int, seed: int = 0) -> Tuple[str, str]:
    """Señas sintéticas con señal de clase separable (para que el modelo pueda aprender)."""
    rng = np.random.default_rng(seed)
    feat_dir = os.path.join(root, "features")
    os.makedirs(feat_dir, exist_ok=True)
    manifest = os.path.join(root, "all.tsv")
    with open(manifest, "w", encoding="utf-8") as f:
        for c in range(classes):
            angle = 2 * np.pi * c / classes
            for k in range(n_per_class):
                x = pc.synthetic_sequence(rng)
                # firma de clase: desplazamiento global de las manos en una dirección
                x = x.reshape(pc.T_ENC, pc.N_POINTS, 3)
                x[:, pc.SL_HANDS, 0] += 0.35 * np.cos(angle)
                x[:, pc.SL_HANDS, 1] += 0.35 * np.sin(angle)
                x = x.reshape(pc.T_ENC, pc.DIM) + rng.normal(0, 0.02, (pc.T_ENC, pc.DIM))
                name = f"c{c:02d}_{k:03d}"
                np.save(os.path.join(feat_dir, f"{name}.npy"), x.astype(np.float32))
                f.write(f"{name}\t{pc.T_ENC}\tsign_{c:02d}\n")
    return manifest, feat_dir


def split_manifest(manifest: str, out_dir: str, val_frac: float, seed: int) -> Tuple[str, str]:
    """Split **estratificado** por clase: un shuffle global deja clases sin ejemplos en train."""
    rows = [l.rstrip("\n") for l in open(manifest, encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    has_group = all(len(r.split("\t")) >= 4 for r in rows)
    tr_rows, va_rows = [], []
    if has_group:
        # split por firmante: validar con firmantes nunca vistos, no con repeticiones
        # del mismo firmante (eso infla la accuracy y no dice nada del producto).
        groups = sorted({r.split("\t")[3] for r in rows})
        n_val = max(1, int(round(len(groups) * val_frac)))
        val_groups = set(groups[:n_val]) if len(groups) > 1 else set()
        for r in rows:
            (va_rows if r.split("\t")[3] in val_groups else tr_rows).append(r)
        print(f"[i] split por firmante: val={sorted(val_groups)}")
    else:
        by_class: dict = {}
        for r in rows:
            by_class.setdefault(r.split("\t")[2], []).append(r)
        for label in sorted(by_class):
            items = by_class[label]
            rng.shuffle(items)
            k = max(1, int(round(len(items) * val_frac))) if len(items) >= 5 else 0
            va_rows += items[:k]
            tr_rows += items[k:]
    if not tr_rows or not va_rows:
        raise SystemExit("split vacío: hacen falta al menos ~5 ejemplos por clase")
    os.makedirs(out_dir, exist_ok=True)
    tr, va = os.path.join(out_dir, "train.tsv"), os.path.join(out_dir, "val.tsv")
    open(tr, "w", encoding="utf-8").write("\n".join(tr_rows) + "\n")
    open(va, "w", encoding="utf-8").write("\n".join(va_rows) + "\n")
    return tr, va


# --------------------------------------------------------------------------- modelo
class PoseEncoderClassifier(nn.Module):
    """Mismos submódulos y nombres que el seq2seq del fork, para poder transferir."""

    def __init__(self, sgn_dim=pc.DIM, hidden=256, ff=2048, layers=6, heads=8,
                 dropout=0.1, n_classes=50):
        super().__init__()
        self.sgn_embed = SpatialEmbeddings(
            embedding_dim=hidden, input_size=sgn_dim, num_heads=heads,
            scale=True, norm_type=None, activation_type=None,
        )
        self.encoder = TransformerEncoder(
            hidden_size=hidden, ff_size=ff, num_layers=layers, num_heads=heads,
            dropout=dropout, emb_dropout=dropout,
        )
        self.classifier = nn.Linear(hidden, n_classes)
        self.hidden = hidden

    def forward(self, sgn, sgn_mask, sgn_lengths):
        out = self.encoder(embed_src=self.sgn_embed(sgn, sgn_mask),
                           src_length=sgn_lengths, mask=sgn_mask)
        final = out[0]                                    # (B, T, H)
        m = sgn_mask.transpose(1, 2).float()              # (B, T, 1)
        pooled = (final * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.classifier(pooled)


# --------------------------------------------------------------------------- métricas
def macro_f1(y_true, y_pred, n_classes) -> float:
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss = 0, 0, 0.0
    y_true, y_pred = [], []
    for sgn, mask, lengths, y in loader:
        sgn, mask, lengths, y = sgn.to(device), mask.to(device), lengths.to(device), y.to(device)
        logits = model(sgn, mask, lengths)
        loss += float(crit(logits, y))
        pred = logits.argmax(-1)
        correct += int((pred == y).sum())
        total += y.numel()
        y_true += y.tolist()
        y_pred += pred.tolist()
    n_cls = model.classifier.out_features
    return correct / max(total, 1), macro_f1(y_true, y_pred, n_cls), loss / max(total, 1)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--feat-dir")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/classifier")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic or not args.manifest:
        manifest, feat_dir = make_synthetic("data/synth_clf", args.per_class, args.classes, args.seed)
        print(f"[i] datos sintéticos: {args.classes} clases x {args.per_class} = "
              f"{args.classes * args.per_class} secuencias")
    else:
        manifest, feat_dir = args.manifest, args.feat_dir
        print(f"[i] manifest: {manifest} | features: {feat_dir}")

    tr_m, va_m = split_manifest(manifest, os.path.join(os.path.dirname(manifest) or ".", "split"),
                                args.val_frac, args.seed)
    tr = PoseClfDataset(tr_m, feat_dir)
    va = PoseClfDataset(va_m, feat_dir)
    n_classes = len(tr.labels)
    if va.labels != tr.labels:
        raise SystemExit("el split dejó clases fuera de train; usa más datos o cambia la semilla")
    print(f"[i] train {len(tr)} | val {len(va)} | clases {n_classes}")

    tr_dl = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate_clf)
    va_dl = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=collate_clf)

    model = PoseEncoderClassifier(n_classes=n_classes, layers=args.layers,
                                  dropout=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[i] encoder+clasificador: {n_par:,} params (encoder {args.layers} capas, hidden 256)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    os.makedirs(args.out, exist_ok=True)
    best_acc, history = -1.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for sgn, mask, lengths, y in tr_dl:
            sgn, mask, lengths, y = sgn.to(device), mask.to(device), lengths.to(device), y.to(device)
            logits = model(sgn, mask, lengths)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss) * y.numel()
        tr_loss = run / len(tr)
        acc, f1, va_loss = evaluate(model, va_dl, device)
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_loss": round(va_loss, 4), "val_acc": round(acc, 4),
                        "val_f1": round(f1, 4)})
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"ep {epoch:3d} | train {tr_loss:6.3f} | val {va_loss:6.3f} "
                  f"| acc {acc:.3f} | F1 {f1:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "labels": tr.labels,
                        "sgn_dim": pc.DIM, "hidden": model.hidden,
                        "epoch": epoch, "val_acc": acc, "val_f1": f1},
                       os.path.join(args.out, "best.pt"))
            enc = {k: v for k, v in model.state_dict().items()
                   if k.startswith("sgn_embed.") or k.startswith("encoder.")}
            torch.save(enc, os.path.join(args.out, "encoder_only.pt"))

    torch.save({"model": model.state_dict(), "labels": tr.labels},
               os.path.join(args.out, "last.pt"))
    json.dump(history, open(os.path.join(args.out, "history.json"), "w"), indent=1)

    chance = 1.0 / n_classes
    print(f"[i] mejor val acc {best_acc:.3f} (azar {chance:.3f}) | checkpoints en {args.out}/")
    print(f"[i] encoder transferible: {args.out}/encoder_only.pt "
          f"({os.path.getsize(os.path.join(args.out, 'encoder_only.pt')) / 1e6:.1f} MB)")
    assert best_acc > chance * 2, f"no aprendió: {best_acc:.3f} vs azar {chance:.3f}"
    print("[PASS] encoder pre-entrenado sobre señas aisladas")


if __name__ == "__main__":
    main()
