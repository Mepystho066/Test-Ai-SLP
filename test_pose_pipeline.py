#!/usr/bin/env python3
"""Regresiones del contrato de pose y del collate. Sin pytest: `python test_pose_pipeline.py`.

Cubre los tres fallos que aparecieron al cablear el Hito 0:
  1. normalización con hombros mal indexados (|x| explota),
  2. máscara inferida del valor cero (frames nulos se pierden como padding),
  3. máscara en float (transformer_layers hace `~mask` y revienta).
"""
import os
import sys
import tempfile
import types

import importlib.machinery


def _stub(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = m
    return m


if "tensorflow" not in sys.modules:
    tf = _stub("tensorflow")
    tf.config = types.SimpleNamespace(set_visible_devices=lambda *a, **k: None)
    tf.nn = types.SimpleNamespace()
    tt = _stub("torchtext")
    ttd = _stub("torchtext.data")
    for n in ("Dataset", "Field", "RawField", "Example", "BucketIterator", "Iterator"):
        setattr(ttd, n, object)
    tt.data = ttd

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pose_contract as pc  # noqa: E402
from pose_dataset import PoseManifest, collate  # noqa: E402
from signjoey.vocabulary import GlossVocabulary, TextVocabulary  # noqa: E402


def test_contract_shapes():
    rng = np.random.default_rng(1)
    x = pc.synthetic_sequence(rng)
    assert x.shape == (pc.T_ENC, pc.DIM) == (128, 318), x.shape
    assert np.isfinite(x).all()
    # normalizado por anchura de hombros: no debe explotar
    assert np.abs(x).max() < 10, np.abs(x).max()
    print(f"[ok] contrato: {x.shape} max|x|={np.abs(x).max():.2f}")


def test_normalization_invariant_to_translation_and_scale():
    rng = np.random.default_rng(2)
    pts = rng.normal(0, 0.02, size=(8, pc.N_POINTS, 3)).astype(np.float32)
    pts[:, pc.SH_L] = (-0.30, 0.42, 0.0)
    pts[:, pc.SH_R] = (-0.10, 0.42, 0.0)      # anchura 0.20
    a = pc.normalize_sequence(pts)
    shifted = pts + np.array([3.0, -2.0, 0.5], dtype=np.float32)   # traslación
    scaled = (pts - np.array([-0.2, 0.42, 0], dtype=np.float32)) * 2.5 + np.array([-0.2, 0.42, 0], dtype=np.float32)
    b = pc.normalize_sequence(shifted)
    c = pc.normalize_sequence(scaled)
    assert np.allclose(a, b, atol=1e-5), "no es invariante a traslación"
    assert np.allclose(a, c, atol=1e-5), "no es invariante a escala"
    print("[ok] normalización invariante a traslación y escala")


def test_mask_keeps_zero_frames():
    """Un frame con todos los valores a 0 es dato, no padding: la máscara sale de lengths."""
    tmp = tempfile.mkdtemp()
    feats = os.path.join(tmp, "features")
    os.makedirs(feats)
    x = np.zeros((10, pc.DIM), dtype=np.float32)
    x[0] = 0.1
    x[5] = 0.0                     # frame nulo en medio de la secuencia
    x[9] = 0.2
    np.save(os.path.join(feats, "a.npy"), x)
    manifest = os.path.join(tmp, "train.tsv")
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("a\t10\thola gracias\nb\t6\tagua casa\n")
    np.save(os.path.join(feats, "b.npy"), np.full((6, pc.DIM), 0.3, dtype=np.float32))

    txt_vocab = TextVocabulary(tokens=["hola", "gracias", "agua", "casa"])
    gls_vocab = GlossVocabulary(tokens=["HOLA", "GRACIAS"])
    ds = PoseManifest(manifest, txt_vocab, gls_vocab, feat_dir=feats, t_max=10)
    batch = collate([ds[0], ds[1]])

    assert batch.sgn_mask.dtype == torch.bool, batch.sgn_mask.dtype
    assert batch.sgn_mask.shape == (2, 1, 10), batch.sgn_mask.shape
    assert int(batch.sgn_mask[0, 0].sum()) == 10, "se perdió el frame nulo"
    assert int(batch.sgn_mask[1, 0].sum()) == 6, "padding mal calculado"
    print(f"[ok] máscara por longitudes (bool, {tuple(batch.sgn_mask.shape)})")


if __name__ == "__main__":
    test_contract_shapes()
    test_normalization_invariant_to_translation_and_scale()
    test_mask_keeps_zero_frames()
    print("[PASS] regresiones del contrato de pose")
