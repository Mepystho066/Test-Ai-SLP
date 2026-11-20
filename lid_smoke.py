#!/usr/bin/env python3
"""Hito 1.5 — smoke test de la cabeza Sign2LID (paper §3.2).

Valida que la cabeza LID (CTC sobre la capa inicial del encoder, ~2,5k params)
se arma y entrena: `lid_vocab_size` clases, pérdida LID finita y backward.
No valida calidad (landmarks sintéticos, pesos aleatorios).

Correr desde la raíz del repo:
    .venv/bin/python lid_smoke.py
"""
import sys
import types
import importlib.machinery


def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__loader__ = None
    sys.modules[name] = mod
    return mod


# El fork aún importa torchtext.data en helpers.py/vocabulary.py (aunque el
# smoke test no lo use). TF ya no es necesario (model.py no lo importa), pero
# el stub se conserva por si algún import transitivo lo toca.
_tf = _stub("tensorflow")
_tf.config = types.SimpleNamespace(set_visible_devices=lambda *a, **k: None)
_tf.nn = types.SimpleNamespace()

_tt = _stub("torchtext")
_tt_data = _stub("torchtext.data")
for _n in ("Dataset", "Field", "RawField", "Example", "BucketIterator", "Iterator"):
    setattr(_tt_data, _n, object)
_tt.data = _tt_data

import torch  # noqa: E402
from signjoey.model import build_model  # noqa: E402
from signjoey.batch import Batch  # noqa: E402
from signjoey.vocabulary import GlossVocabulary, TextVocabulary  # noqa: E402
from signjoey.loss import XentLoss  # noqa: E402


def main():
    torch.manual_seed(0)
    D, T, B = 318, 128, 2
    H = 256
    N_LID = 4  # blank + 3 lenguas de señas (ej. gsg, csl, lsc)

    gls_vocab = GlossVocabulary(tokens=["G_A", "G_B", "G_C", "G_D", "G_E"])
    txt_vocab = TextVocabulary(tokens=[f"W{i}" for i in range(4, 12)])

    cfg_model = {
        "tied_softmax": False,
        "encoder": {"type": "transformer", "hidden_size": H, "ff_size": 2048,
                    "num_layers": 6, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": H, "dropout": 0.1, "scale": True}},
        "decoder": {"type": "transformer", "hidden_size": H, "ff_size": 2048,
                    "num_layers": 6, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": H, "dropout": 0.1, "scale": True}},
    }
    model = build_model(cfg=cfg_model, gls_vocab=gls_vocab, txt_vocab=txt_vocab,
                        sgn_dim=D, do_recognition=True, do_joint=True,
                        do_translation=True, lid_vocab_size=N_LID)
    assert model.do_lid, "la cabeza LID no se armó"
    lid_params = (model.lid_output_layer.weight.numel()
                  + model.lid_output_layer.bias.numel())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] LID head: {N_LID} clases, {lid_params} params "
          f"({lid_params / 1000:.1f}k) | modelo total {n_params:,}")

    # --- batch sintético con LIDtok ---
    tb = types.SimpleNamespace()
    tb.sequence = ["seq1", "seq2"]
    tb.signer = ["s1", "s2"]
    sgn = torch.zeros(B, T, D)
    sgn[0] = torch.randn(T, D)
    sgn[1, :64] = torch.randn(64, D)
    tb.sgn = (sgn, torch.tensor([128, 64]))

    txt = torch.full((B, 6), 1, dtype=torch.long)
    txt[0, :5] = torch.tensor([2, 4, 5, 6, 3])
    txt[1, :4] = torch.tensor([2, 7, 8, 3])
    tb.txt = (txt, torch.tensor([5, 4]))

    gls = torch.full((B, 4), 2, dtype=torch.long)
    gls[0, :3] = torch.tensor([3, 4, 5])
    gls[1, :2] = torch.tensor([3, 4])
    tb.gls = (gls, torch.tensor([3, 2]))

    # LIDtok (paper §3.2): longitud = texto hablado; cada posición = id de la
    # lengua de señas (1..N, 0 = blank del CTC). seq1 -> lengua 1, seq2 -> lengua 2.
    lid = torch.zeros(B, 6, dtype=torch.long)
    lid[0, :5] = 1
    lid[1, :4] = 2
    tb.lid = (lid, torch.tensor([5, 4]))

    batch = Batch(is_train=True, torch_batch=tb,
                  txt_pad_index=model.txt_pad_index, sgn_dim=D, use_cuda=False)
    assert batch.lid is not None and tuple(batch.lid.shape) == (B, 6)

    rec_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    joint_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    trans_fn = XentLoss(pad_index=model.txt_pad_index, smoothing=0.0)
    lid_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    rec, joint, trans, lid_loss = model.get_loss_for_batch(
        batch=batch, recognition_loss_function=rec_fn,
        joint_loss_function=joint_fn, translation_loss_function=trans_fn,
        recognition_loss_weight=1.0, joint_loss_weight=5.0,
        translation_loss_weight=3.0, lid_loss_function=lid_fn,
        lid_loss_weight=1.0,
    )
    print(f"[OK] losses  rec={rec.item():.4f}  joint={joint.item():.4f}  "
          f"trans={trans.item():.4f}  lid={lid_loss.item():.4f}")

    assert lid_loss is not None and torch.isfinite(lid_loss)
    (rec + joint + trans + lid_loss).backward()
    print("[PASS] Sign2LID smoke test superado")


if __name__ == "__main__":
    main()
