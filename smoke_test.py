#!/usr/bin/env python3
"""Hito 0 — smoke test del fork Tan et al. (SignJoey) con encoder de pose.

Valida el cableado del modelo (recognition CTC + joint TxtCTC + translation
attention) sobre tensores sintéticos (B=2, T=128, D=318) SIN pasar por el
data loader (dataset.py está hardcodeado a CSL-Daily) ni por TensorFlow
(solo se usa en el beam search CTC) ni por torchtext.data (API legacy).

Correr desde la raíz del repo, con torch instalado:
    python smoke_test.py
"""
import sys
import types

# --- stubs: el fork importa tensorflow (model.py) y torchtext.data
# (helpers.py, vocabulary.py) a nivel de módulo, aunque el smoke test
# no los use. Sin estos stubs el import de signjoey.model revienta. ---
_tf = types.ModuleType("tensorflow")
_tf.config = types.SimpleNamespace(set_visible_devices=lambda *a, **k: None)
_tf.nn = types.SimpleNamespace()
sys.modules["tensorflow"] = _tf

_tt = types.ModuleType("torchtext")
_tt_data = types.ModuleType("torchtext.data")
_tt_data.Dataset = object
_tt_data.Field = object
_tt_data.RawField = object
_tt_data.Example = object
_tt_data.BucketIterator = object
_tt_data.Iterator = object
_tt.data = _tt_data
sys.modules["torchtext"] = _tt
sys.modules["torchtext.data"] = _tt_data

import torch  # noqa: E402
from signjoey.model import build_model  # noqa: E402
from signjoey.batch import Batch  # noqa: E402
from signjoey.vocabulary import GlossVocabulary, TextVocabulary  # noqa: E402
from signjoey.loss import XentLoss  # noqa: E402


def main():
    torch.manual_seed(0)
    D, T, B = 318, 128, 2
    H = 256

    # Vocabularios mínimos. TextVocabulary reserva 0=UNK,1=PAD,2=BOS,3=EOS.
    gls_vocab = GlossVocabulary(tokens=["G_A", "G_B", "G_C", "G_D", "G_E"])
    txt_vocab = TextVocabulary(tokens=[f"W{i}" for i in range(4, 12)])

    cfg_model = {
        "tied_softmax": False,
        "encoder": {
            "type": "transformer", "hidden_size": H, "ff_size": 2048,
            "num_layers": 6, "num_heads": 8, "dropout": 0.1,
            "embeddings": {"embedding_dim": H, "dropout": 0.1, "scale": True},
        },
        "decoder": {
            "type": "transformer", "hidden_size": H, "ff_size": 2048,
            "num_layers": 6, "num_heads": 8, "dropout": 0.1,
            "embeddings": {"embedding_dim": H, "dropout": 0.1, "scale": True},
        },
    }

    model = build_model(
        cfg=cfg_model, gls_vocab=gls_vocab, txt_vocab=txt_vocab,
        sgn_dim=D, do_recognition=True, do_joint=True, do_translation=True,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] model built, trainable params = {n_params:,}")

    # --- batch sintético (contrato: B=2, T=128, D=318 + máscara) ---
    tb = types.SimpleNamespace()
    tb.sequence = ["seq1", "seq2"]
    tb.signer = ["s1", "s2"]

    sgn = torch.zeros(B, T, D)
    sgn[0] = torch.randn(T, D)
    sgn[1, :64] = torch.randn(64, D)
    sgn_lengths = torch.tensor([128, 64])
    tb.sgn = (sgn, sgn_lengths)

    # TextVocabulary: 0=UNK,1=PAD,2=BOS,3=EOS; tokens válidos desde 4 (evita blank).
    txt = torch.full((B, 6), 1, dtype=torch.long)  # pad
    txt[0, :5] = torch.tensor([2, 4, 5, 6, 3])     # BOS w w w EOS
    txt[1, :4] = torch.tensor([2, 7, 8, 3])
    txt_lengths = torch.tensor([5, 4])
    tb.txt = (txt, txt_lengths)

    # GlossVocabulary: 0=SIL,1=UNK,2=PAD; tokens válidos desde 3 (evita blank).
    gls = torch.full((B, 4), 2, dtype=torch.long)  # pad
    gls[0, :3] = torch.tensor([3, 4, 5])
    gls[1, :2] = torch.tensor([3, 4])
    gls_lengths = torch.tensor([3, 2])
    tb.gls = (gls, gls_lengths)

    batch = Batch(is_train=True, torch_batch=tb,
                  txt_pad_index=model.txt_pad_index, sgn_dim=D, use_cuda=False)
    print(f"[OK] batch: sgn {tuple(batch.sgn.shape)} mask {tuple(batch.sgn_mask.shape)}")

    rec_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    joint_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    trans_fn = XentLoss(pad_index=model.txt_pad_index, smoothing=0.0)

    rec, joint, trans, lid = model.get_loss_for_batch(
        batch=batch,
        recognition_loss_function=rec_fn,
        joint_loss_function=joint_fn,
        translation_loss_function=trans_fn,
        recognition_loss_weight=1.0,
        joint_loss_weight=5.0,
        translation_loss_weight=3.0,
    )
    print(f"[OK] losses  rec={rec.item():.4f}  joint={joint.item():.4f}  "
          f"trans={trans.item():.4f}  lid={lid}")

    total = rec + joint + trans
    total.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[OK] backward ok, {grads} tensores con gradiente")
    print("[PASS] smoke test superado")


if __name__ == "__main__":
    main()
