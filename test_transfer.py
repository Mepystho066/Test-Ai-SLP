#!/usr/bin/env python3
"""Comprueba que el encoder pre-entrenado por `train.py` entra en el seq2seq del fork.

Es la promesa de todo el Hito A: si las claves no coinciden, el checkpoint de
clasificación no sirve para traducir y habría que re-entrenar el encoder desde cero.
"""
import importlib.machinery
import sys
import types

try:
    import torchtext  # noqa: F401
except ImportError:
    _tt = types.ModuleType("torchtext")
    _tt.__spec__ = importlib.machinery.ModuleSpec("torchtext", loader=None)
    _ttd = types.ModuleType("torchtext.data")
    _ttd.__spec__ = importlib.machinery.ModuleSpec("torchtext.data", loader=None)
    for _n in ("Dataset", "Field", "RawField", "Example", "BucketIterator", "Iterator"):
        setattr(_ttd, _n, object)
    _tt.data = _ttd
    sys.modules["torchtext"], sys.modules["torchtext.data"] = _tt, _ttd

import torch  # noqa: E402

import pose_contract as pc  # noqa: E402
from signjoey.model import build_model  # noqa: E402
from signjoey.vocabulary import GlossVocabulary, TextVocabulary  # noqa: E402

CKPT = "models/classifier/encoder_only.pt"   # override: python test_transfer.py <ruta>


def main():
    global CKPT
    if len(sys.argv) > 1:
        CKPT = sys.argv[1]
    txt = TextVocabulary(tokens=["hola", "agua", "gracias", "hoy", "frio"])
    gls = GlossVocabulary(tokens=["HOLA", "AGUA", "GRACIAS"])
    enc_state = torch.load(CKPT, map_location="cpu")
    n_layers = 1 + max(int(k.split(".")[2]) for k in enc_state if k.startswith("encoder.layers."))
    print(f"checkpoint: {CKPT} ({n_layers} capas de encoder)")
    cfg = {
        "tied_softmax": True,
        "encoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": n_layers, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.1, "scale": True}},
        "decoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": 6, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.1, "scale": True}},
    }
    model = build_model(cfg=cfg, gls_vocab=gls, txt_vocab=txt, sgn_dim=pc.DIM,
                        do_recognition=True, do_joint=True, do_translation=True,
                        lid_vocab_size=0)

    target = model.state_dict()
    unexpected = [k for k in enc_state if k not in target]
    matched = [k for k in enc_state if k in target]
    shapes_bad = [k for k in matched if enc_state[k].shape != target[k].shape]

    print(f"pesos del encoder pre-entrenado : {len(enc_state)}")
    print(f"coinciden en el seq2seq         : {len(matched)}")
    print(f"sobran (no están en el modelo)  : {len(unexpected)} {unexpected[:3]}")
    print(f"formas incompatibles            : {len(shapes_bad)} {shapes_bad[:3]}")
    assert not unexpected, f"claves huérfanas: {unexpected[:5]}"
    assert not shapes_bad, f"formas distintas: {shapes_bad[:5]}"

    missing, unexpected2 = model.load_state_dict(enc_state, strict=False)
    enc_prefixes = ("sgn_embed.", "encoder.")
    missing_enc = [k for k in missing if k.startswith(enc_prefixes)]
    print(f"faltantes que tocan al encoder  : {len(missing_enc)}")
    assert not missing_enc, f"el encoder no quedó completo: {missing_enc[:5]}"

    # el decoder y las cabezas siguen sin entrenar a propósito
    print(f"módulos que siguen aleatorios   : {len(missing)} (decoder/cabezas)")
    print("[PASS] el encoder pre-entrenado se transfiere al seq2seq")


if __name__ == "__main__":
    main()
