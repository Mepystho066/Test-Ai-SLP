#!/usr/bin/env python3
"""Hito 1 — decodificación sin TensorFlow.

Prueba, sobre una secuencia sintética del contrato:
  1. `ctc_greedy_decode` y `ctc_prefix_beam_search` (signjoey/ctc_decode.py),
  2. `model.run_batch` con `do_recognition=True` -> usa el CTC de glosas en torch,
  3. greedy de traducción y el beam search conjunto del fork,
  4. `signjoey/joint_decode.py` (beam de atención + rescoring CTC correcto).

Valida cableado y contrato de salida, NO calidad: los pesos son aleatorios y los
landmarks sintéticos. El decode real necesita features de PHOENIX (ver @gea).

Correr:
    .venv/bin/python decode_smoke.py
"""
import importlib.machinery
import sys
import types

# torchtext sigue haciendo falta (vocabulary.py/helpers.py lo importan); TensorFlow ya no.
assert "tensorflow" not in sys.modules


def _stub(name):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = m
    return m


_tt = _stub("torchtext")
_ttd = _stub("torchtext.data")
for _n in ("Dataset", "Field", "RawField", "Example", "BucketIterator", "Iterator"):
    setattr(_ttd, _n, object)
_tt.data = _ttd

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pose_contract as pc  # noqa: E402
from pose_dataset import PoseManifest, collate  # noqa: E402
from signjoey.ctc_decode import (  # noqa: E402
    ctc_greedy_decode, ctc_prefix_beam_search, log_softmax_frames,
)
from signjoey.joint_decode import joint_decode  # noqa: E402
from signjoey.model import build_model  # noqa: E402
from signjoey.vocabulary import GlossVocabulary, TextVocabulary  # noqa: E402

WORDS = ["hola", "gracias", "agua", "hoy", "frio"]


def build(seed=0, dim_head=64):
    torch.manual_seed(seed)
    txt_vocab = TextVocabulary(tokens=WORDS)
    # head inflada a dim_head para simular el vocab mT5 recortado (64k) sin el coste real
    gls_vocab = GlossVocabulary(tokens=[w.upper() for w in WORDS])
    cfg = {
        "tied_softmax": False,
        "encoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": 2, "num_heads": 8, "dropout": 0.0,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.0, "scale": True}},
        "decoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": 2, "num_heads": 8, "dropout": 0.0,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.0, "scale": True}},
    }
    model = build_model(cfg=cfg, gls_vocab=gls_vocab, txt_vocab=txt_vocab,
                        sgn_dim=pc.DIM, do_recognition=True, do_joint=True,
                        do_translation=True)
    model.eval()
    return model, txt_vocab, gls_vocab


def make_batch(model, txt_vocab, n=2):
    rng = np.random.default_rng(0)
    t = torch.tensor
    feats = torch.stack([torch.from_numpy(pc.synthetic_sequence(rng)) for _ in range(n)])
    lens = t([pc.T_ENC, pc.T_ENC // 2])
    txt = t([[2, 4, 5, 3], [2, 5, 3, 1]])   # el segundo va con pad al final
    gls = t([[3, 4], [4, 5]])
    from types import SimpleNamespace
    tb = SimpleNamespace(
        sequence=[f"s{i}" for i in range(n)], signer=["s"] * n,
        sgn=(feats, lens), txt=(txt, t([4, 3])), gls=(gls, t([2, 2])),
    )
    from signjoey.batch import Batch
    b = Batch(is_train=False, torch_batch=tb, txt_pad_index=1, sgn_dim=pc.DIM)
    b.sgn_mask = (torch.arange(pc.T_ENC).unsqueeze(0) < lens.unsqueeze(1)).unsqueeze(1)
    return b


def to_text(ids, vocab):
    return " ".join(vocab.itos[i] if i < len(vocab.itos) else f"<{i}>" for i in ids)


def test_ctc_decoders():
    # frame "peaked" en el token 3, con blanks alrededor: greedy y beam deben dar [3]
    T, V = 5, 6
    lp = torch.full((T, 1, V), -10.0)
    lp[:, 0, 0] = 0.0          # blank
    lp[2, 0, 0] = -10.0        # el frame central no es blank...
    lp[2, 0, 3] = 0.0          # ...es el token 3
    g = ctc_greedy_decode(lp, torch.tensor([T]))[0]
    b = ctc_prefix_beam_search(lp, torch.tensor([T]), beam_width=4)[0]
    assert g == [3], g
    assert b == [3], b
    # repetición colapsada: 3,3 -> [3]
    lp2 = torch.full((4, 1, V), -10.0)
    lp2[:, 0, 0] = 0.0
    lp2[[1, 2], 0, 0] = -10.0
    lp2[[1, 2], 0, 4] = 0.0
    assert ctc_greedy_decode(lp2, torch.tensor([4]))[0] == [4]
    assert ctc_prefix_beam_search(lp2, torch.tensor([4]), beam_width=4)[0] == [4]
    print("[ok] CTC greedy + prefix beam (colapso y blank)")


def main():
    test_ctc_decoders()
    model, txt_vocab, gls_vocab = build()
    batch = make_batch(model, txt_vocab)

    # --- 1. run_batch: reconocimiento CTC (torch) + greedy de traducción ---
    gls_out, txt_out, _ = model.run_batch(
        batch, recognition_beam_size=4, translation_beam_size=1,
        translation_max_output_length=12,
    )
    print("[ok] run_batch sin TF")
    print("     glosas decodificadas:", [to_text(g, gls_vocab) for g in gls_out])
    print("     texto greedy        :", [to_text(t, txt_vocab) for t in txt_out])

    # --- 2. beam search conjunto del fork ---
    _, txt_joint, _ = model.run_batch(
        batch, recognition_beam_size=1, translation_beam_size=4,
        translation_beam_alpha=0.0, translation_max_output_length=12,
    )
    print("[ok] joint_beam_search del fork")
    print("     texto joint (fork)  :", [to_text(t, txt_vocab) for t in txt_joint])
    assert all(all(0 <= int(i) < len(txt_vocab) for i in h) for h in txt_joint)

    # --- 3. joint_decode.py (atención + rescoring CTC) ---
    enc_out, enc_hid, spec, _first = model.encode(batch.sgn, batch.sgn_mask, batch.sgn_lengths)
    hyps, scores = joint_decode(
        decoder=model.decoder, embed=model.txt_embed,
        encoder_output=enc_out, encoder_hidden=enc_hid, src_mask=batch.sgn_mask,
        bos_index=model.txt_bos_index, eos_index=model.txt_eos_index,
        pad_index=model.txt_pad_index, max_output_length=12,
        ctc_layer=model.text_output_layer, ctc_weight=0.3, beam=4,
    )
    print("[ok] joint_decode (atención + CTC)")
    print("     texto joint (nuevo):", [to_text(h, txt_vocab) for h in hyps])
    print("     scores             :", [round(s, 3) for s in scores])
    assert len(hyps) == batch.sgn.size(0)
    assert all(model.txt_bos_index not in h for h in hyps)

    assert "tensorflow" not in sys.modules, "algo importó TensorFlow"
    print("[PASS] decodificación sin TensorFlow")


if __name__ == "__main__":
    main()
