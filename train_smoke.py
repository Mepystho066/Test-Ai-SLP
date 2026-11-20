#!/usr/bin/env python3
"""Hito 0.5 — entrenamiento mínimo sobre datos del contrato (pose 318 / T_enc 128).

Qué prueba (y qué no):
  SÍ: que el modelo del fork entrena con features de pose del contrato — shape, máscara
      por longitudes reales, tres pérdidas (recognition CTC + joint TxtCTC + attention),
      backward, clip de gradiente, un paso de Adam y checkpoint.
  NO: que aprenda algo útil. Los landmarks son sintéticos (sin coarticulación, sin
      gramática facial), el vocabulario es de juguete y las frases son aleatorias.

Correr:
    .venv/bin/python train_smoke.py --steps 40 --batch 4
"""
import argparse
import os
import sys
import time
import types

# --- stubs de imports legacy (ver smoke_test.py) ---
# torch>=2 importa torch._dynamo desde el optimizador, y dynamo llama a
# importlib.util.find_spec() sobre los módulos del sys.path: un ModuleType falso
# sin __spec__ revienta con "tensorflow.__spec__ is None". Hay que darle un spec.
import importlib.machinery  # noqa: E402


def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__loader__ = None
    sys.modules[name] = mod
    return mod


_tf = _stub("tensorflow")
_tf.config = types.SimpleNamespace(set_visible_devices=lambda *a, **k: None)
_tf.nn = types.SimpleNamespace()

_tt = _stub("torchtext")
_tt_data = _stub("torchtext.data")
for _n in ("Dataset", "Field", "RawField", "Example", "BucketIterator", "Iterator"):
    setattr(_tt_data, _n, object)
_tt.data = _tt_data

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import pose_contract as pc  # noqa: E402
from pose_dataset import PoseManifest, collate  # noqa: E402
from signjoey.model import build_model  # noqa: E402
from signjoey.vocabulary import GlossVocabulary, TextVocabulary  # noqa: E402
from signjoey.loss import XentLoss  # noqa: E402

WORDS = ["hola", "gracias", "agua", "casa", "trabajo", "ayer", "mañana", "fiebre",
         "dolor", "ayuda", "familia", "comer", "ir", "querer", "nombre", "cuanto"]


def make_dataset(root: str, n: int, seed: int = 0) -> str:
    """Escribe n secuencias .npy del contrato + un manifest TSV."""
    rng = np.random.default_rng(seed)
    feat_dir = os.path.join(root, "features")
    os.makedirs(feat_dir, exist_ok=True)
    manifest = os.path.join(root, "train.tsv")
    with open(manifest, "w", encoding="utf-8") as f:
        for i in range(n):
            x = pc.synthetic_sequence(rng)          # (T_ENC, 318)
            np.save(os.path.join(feat_dir, f"syn{i:03d}.npy"), x)
            k = rng.integers(3, 7)
            text = " ".join(rng.choice(WORDS, size=k, replace=False))
            f.write(f"syn{i:03d}\t{x.shape[0]}\t{text}\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    manifest = make_dataset(args.root, args.n, args.seed)
    texts = [l.rstrip("\n").split("\t")[2] for l in open(manifest, encoding="utf-8")]

    vocab_words = sorted({w for t in texts for w in t.split()})
    txt_vocab = TextVocabulary(tokens=vocab_words)
    gls_vocab = GlossVocabulary(tokens=[w.upper() for w in vocab_words])

    ds = PoseManifest(manifest, txt_vocab, gls_vocab,
                      feat_dir=os.path.join(args.root, "features"),
                      t_max=pc.T_ENC)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)

    cfg_model = {
        "tied_softmax": False,
        "encoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": 6, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.1, "scale": True}},
        "decoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": 6, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.1, "scale": True}},
    }
    model = build_model(cfg=cfg_model, gls_vocab=gls_vocab, txt_vocab=txt_vocab,
                        sgn_dim=pc.DIM, do_recognition=True, do_joint=True,
                        do_translation=True)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i] vocab txt={len(txt_vocab)} gls={len(gls_vocab)} | sgn_dim={pc.DIM} "
          f"| T_enc={pc.T_ENC} | params={n_par:,}")

    rec_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    joint_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    trans_fn = XentLoss(pad_index=1, smoothing=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    hist, t0, step, epoch = [], time.time(), 0, 0
    while step < args.steps:
        epoch += 1
        for batch in dl:
            rec, joint, trans, lid = model.get_loss_for_batch(
                batch=batch, recognition_loss_function=rec_fn,
                translation_loss_function=trans_fn, joint_loss_function=joint_fn,
                recognition_loss_weight=1.0, joint_loss_weight=5.0,
                translation_loss_weight=3.0,
            )
            loss = rec + joint + trans
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            hist.append((loss.item(), rec.item(), joint.item(), trans.item()))
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step:3d} | loss {hist[-1][0]:8.2f} | rec {hist[-1][1]:7.2f} "
                      f"| joint {hist[-1][2]:7.2f} | trans {hist[-1][3]:7.2f}")
            if step >= args.steps:
                break

    os.makedirs("models/smoke", exist_ok=True)
    ckpt = "models/smoke/pose_smoke.pt"
    torch.save({"model": model.state_dict(), "sgn_dim": pc.DIM, "t_enc": pc.T_ENC}, ckpt)

    first = np.mean([h[0] for h in hist[:5]])
    last = np.mean([h[0] for h in hist[-5:]])
    dt = time.time() - t0
    print(f"[i] media primeros 5: {first:.2f} | últimos 5: {last:.2f} "
          f"| {dt:.1f}s ({dt/max(step,1)*1000:.0f} ms/step CPU)")
    print(f"[i] checkpoint: {ckpt}")
    assert np.isfinite(last), "loss no finita"
    assert last < first, f"la pérdida no bajó ({first:.2f} -> {last:.2f})"
    print("[PASS] entrenamiento mínimo sobre contrato de pose")


if __name__ == "__main__":
    main()
