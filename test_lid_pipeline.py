#!/usr/bin/env python3
"""Verificación del cableado LID end-to-end: manifest -> collate -> modelo -> pérdida.

Complementa a lid_smoke.py (que arma el Batch a mano): aquí el `Batch` sale del
loader real (`PoseManifest` + `collate`) para comprobar que el LIDtok viaja con la
forma y las longitudes que espera `model.get_loss_for_batch`.

Se ejecuta desde la raíz del repo:

    .venv/bin/python test_lid_pipeline.py
"""
import importlib.machinery
import sys
import types

# --- stubs de imports legacy (ver train_smoke.py) ---
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

import torch  # noqa: E402

import pose_contract as pc  # noqa: E402
from pose_dataset import LID_DEFAULT, PoseManifest, collate  # noqa: E402
from signjoey.loss import XentLoss  # noqa: E402
from signjoey.model import build_model  # noqa: E402
from signjoey.vocabulary import GlossVocabulary, TextVocabulary  # noqa: E402

N_LID = 4  # 3 lenguas + blank (0)
PAD = 1


def main() -> int:
    if LID_DEFAULT == 0:
        print("[FAIL] LID_DEFAULT=0 colisiona con el blank del CTC")
        return 1

    root = "data/synth"
    words = set()
    with open(f"{root}/train.tsv", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                words.update(line.split("\t")[2].split())
    words = sorted(words)
    txt_vocab = TextVocabulary(tokens=words)
    gls_vocab = GlossVocabulary(tokens=[w.upper() for w in words])

    ds = PoseManifest(f"{root}/train.tsv", txt_vocab, gls_vocab,
                      feat_dir=f"{root}/features", t_max=pc.T_ENC)
    dl = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate)
    batch = next(iter(dl))

    assert batch.lid is not None, "el Batch no llevó LID"
    # `Batch` desplaza el texto: txt_ori = texto completo, txt = texto[:, 1:],
    # txt_input = texto[:, :-1]. El LIDtok cubre la secuencia hablada completa
    # (BOS..EOS), así que su ancho coincide con txt_ori y sus longitudes con
    # txt_lengths (que `Batch` no recorta).
    assert batch.lid.shape == batch.txt_ori.shape, (batch.lid.shape, batch.txt_ori.shape)
    assert torch.equal(batch.lid_lengths, batch.txt_lengths), "longitudes LID != texto"
    assert int(batch.lid.min()) >= 1, "hay posiciones en blank dentro del target LID"
    print(f"[OK] lid {tuple(batch.lid.shape)} | lengths {batch.lid_lengths.tolist()} "
          f"| ids {torch.unique(batch.lid).tolist()}")

    cfg = {
        "tied_softmax": False,
        "encoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": 2, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.1, "scale": True}},
        "decoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": 2, "num_heads": 8, "dropout": 0.1,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.1, "scale": True}},
    }
    model = build_model(cfg=cfg, gls_vocab=gls_vocab, txt_vocab=txt_vocab,
                        sgn_dim=pc.DIM, do_recognition=True, do_joint=True,
                        do_translation=True, lid_vocab_size=N_LID)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    rec, joint, trans, lid = model.get_loss_for_batch(
        batch=batch, recognition_loss_function=ctc,
        translation_loss_function=XentLoss(pad_index=PAD, smoothing=0.0),
        joint_loss_function=ctc,
        recognition_loss_weight=1.0, joint_loss_weight=5.0,
        translation_loss_weight=3.0, lid_loss_function=ctc, lid_loss_weight=1.0)

    assert lid is not None and torch.isfinite(lid), f"lid_loss inválida: {lid}"
    assert lid.item() > 0, "lid_loss == 0: el target se fue entero al blank"
    print(f"[OK] pérdidas rec={rec.item():.2f} joint={joint.item():.2f} "
          f"trans={trans.item():.2f} lid={lid.item():.4f}")
    print("[PASS] LID end-to-end desde el loader de pose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
