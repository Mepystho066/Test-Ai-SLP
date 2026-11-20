#!/usr/bin/env python3
"""Servidor de inferencia para el fork MLSLT (encoder de pose).

Arranca un modelo SignJoey (encoder transformer + decoder + TxtCTC) sobre el
vocabulario mT5 recortado y expone:

    GET  /             -> webcam_view.html (el widget de cámara)
    POST /predict      -> {"data": [[...318], ...], "shape": [T, 318]}  =>  {"text": ...}
    GET  /health       -> {"status": "ok", "model": ...}

Los pesos son ALEATORIOS: la decodificación es real (beam conjunto atención+CTC
sobre SentencePiece mT5), pero el texto no tendrá sentido hasta entrenar.

Correr (desde la raíz del repo):

    .venv/bin/python infer_server.py --port 8000

y abrir http://localhost:8000
"""
from __future__ import annotations

import argparse
import importlib.machinery
import math
import sys
import time
import types
from pathlib import Path

# --- stubs de imports legacy (ver train_smoke.py) ---
def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mod.__loader__ = None
    sys.modules[name] = mod
    return mod


_tt = _stub("torchtext")
_tt_data = _stub("torchtext.data")
for _n in ("Dataset", "Field", "RawField", "Example", "BucketIterator", "Iterator"):
    setattr(_tt_data, _n, object)
_tt.data = _tt_data

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import pose_contract as pc  # noqa: E402
from pose_vocab import TrimmedVocab  # noqa: E402
from signjoey.joint_decode import joint_decode  # noqa: E402
from signjoey.model import build_model  # noqa: E402
from signjoey.vocabulary import GlossVocabulary  # noqa: E402
from train import PoseEncoderClassifier  # noqa: E402

REPO = Path(__file__).resolve().parent
VOCAB_DIR = REPO.parent / "vocab"
HTML = REPO / "webcam_view.html"


class PredictBody(BaseModel):
    data: list
    shape: list | None = None


def load_model(num_layers: int = 2):
    txt_vocab = TrimmedVocab(
        str(VOCAB_DIR / "mt5_trimmed_64000.model"),
        str(VOCAB_DIR / "mt5_trimmed_64000.map.json"),
    )
    gls_vocab = GlossVocabulary(tokens=["x"])  # do_recognition=False, solo placeholder
    cfg = {
        "tied_softmax": True,
        "encoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": num_layers, "num_heads": 8, "dropout": 0.0,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.0, "scale": True}},
        "decoder": {"type": "transformer", "hidden_size": 256, "ff_size": 2048,
                    "num_layers": num_layers, "num_heads": 8, "dropout": 0.0,
                    "embeddings": {"embedding_dim": 256, "dropout": 0.0, "scale": True}},
    }
    model = build_model(cfg=cfg, sgn_dim=pc.DIM, gls_vocab=gls_vocab,
                        txt_vocab=txt_vocab, do_recognition=False, do_joint=True,
                        do_translation=True)
    model.eval()
    return model, txt_vocab


def load_classifier(path: Path):
    """Carga el checkpoint de clasificación de LSC50 (encoder + cabeza lineal).

    Ese checkpoint SÍ está entrenado (50 señas aisladas, val acc 0.275 con firmantes
    no vistos). Es la única salida que hoy tiene sentido mostrar en la UI.
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck["model"]
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers."))
    clf = PoseEncoderClassifier(sgn_dim=ck.get("sgn_dim", pc.DIM),
                                hidden=ck.get("hidden", 256),
                                layers=n_layers, dropout=0.0,
                                n_classes=len(ck["labels"]))
    clf.load_state_dict(sd)
    clf.eval()
    print(f"[i] clasificador LSC50: {n_layers} capas, {len(ck['labels'])} señas, "
          f"val_acc={ck.get('val_acc')}")
    return clf, list(ck["labels"])


def predict_sign(clf, labels, arr: np.ndarray, topk: int = 3):
    """arr (T, 318) -> [(seña, prob), ...] ordenado desc."""
    sgn = torch.from_numpy(arr).unsqueeze(0)
    mask = torch.ones(1, 1, arr.shape[0], dtype=torch.bool)
    lengths = torch.tensor([arr.shape[0]], dtype=torch.long)
    with torch.no_grad():
        logits = clf(sgn, mask, lengths)
        probs = logits.softmax(-1)[0]
    k = min(topk, probs.numel())
    top = probs.topk(k)
    return [(labels[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


def predict(model, vocab, arr: np.ndarray, beam: int, max_len: int):
    """arr: (T, 318) float32 -> texto decodificado (beam conjunto atención+CTC)."""
    T = arr.shape[0]
    sgn = torch.from_numpy(arr).unsqueeze(0)                     # (1, T, 318)
    mask = torch.ones(1, 1, T, dtype=torch.bool)
    lengths = torch.tensor([T], dtype=torch.long)

    with torch.no_grad():
        enc_out, enc_hid, _spec, _first = model.encode(sgn, mask, lengths)
        hyps, scores = joint_decode(
            decoder=model.decoder, embed=model.txt_embed,
            encoder_output=enc_out, encoder_hidden=enc_hid, src_mask=mask,
            bos_index=model.txt_bos_index, eos_index=model.txt_eos_index,
            pad_index=model.txt_pad_index, max_output_length=max_len,
            ctc_layer=model.text_output_layer, ctc_weight=0.3, beam=beam,
        )
    ids = hyps[0]
    text = vocab.decode(ids)
    return text, ids, float(scores[0])


def build_app() -> FastAPI:
    app = FastAPI(title="MLSLT infer server")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/")
    def index():
        return FileResponse(str(HTML), media_type="text/html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "mlslt-pose",
                "trained": {"classifier_lsc50": app.state.clf is not None,
                            "translation": app.state.seq2seq_trained}}

    @app.post("/predict")
    def predict_endpoint(body: PredictBody):
        try:
            arr = np.asarray(body.data, dtype=np.float32)
            if body.shape is not None:
                arr = arr.reshape(tuple(body.shape))
            if arr.ndim == 2 and arr.shape[1] != pc.DIM:
                return JSONResponse(status_code=400, content={"error": f"dim {arr.shape[1]} != {pc.DIM}"})
            if arr.ndim != 2:
                return JSONResponse(status_code=400, content={"error": f"esperaba 2D, llegó {arr.ndim}D"})
            t0 = time.time()
            out = {"ms": None}

            # 1) salida con sentido: clasificador de señas aisladas (entrenado)
            if app.state.clf is not None:
                top = predict_sign(app.state.clf, app.state.labels, arr)
                out["top3"] = [{"sign": s, "p": round(p, 4)} for s, p in top]
                out["sign"] = top[0][0]
                out["confidence"] = round(top[0][1], 4)
                out["text"] = top[0][0]

            # 2) traducción seña->frase: solo si hay un seq2seq entrenado
            if app.state.seq2seq_trained:
                text, ids, score = predict(app.state.model, app.state.vocab, arr,
                                           app.state.beam, app.state.max_len)
                out["text"] = text
                out["ids"] = ids
                out["score"] = score if math.isfinite(score) else None
            else:
                out["note"] = ("traducción no disponible: el seq2seq no tiene checkpoint "
                               "(hacen falta frases continuas, no señas aisladas)")

            out["ms"] = round((time.time() - t0) * 1000, 1)
            return out
        except Exception as e:  # noqa: BLE001
            return JSONResponse(status_code=500, content={"error": str(e)})

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--clf-ckpt", default="models/lsc50_enc/best.pt",
                    help="checkpoint del clasificador LSC50 (vacío = no usarlo)")
    ap.add_argument("--seq2seq-ckpt", default="",
                    help="checkpoint seq2seq entrenado (opcional; hoy no existe)")
    args = ap.parse_args()

    t0 = time.time()
    model, vocab = load_model(args.layers)
    n = sum(p.numel() for p in model.parameters())
    print(f"[i] modelo cargado: {n:,} params · vocab {len(vocab)} piezas "
          f"· {time.time() - t0:.1f}s")

    clf, labels = (None, [])
    clf_path = REPO / args.clf_ckpt if args.clf_ckpt else None
    if clf_path and clf_path.exists():
        clf, labels = load_classifier(clf_path)

    if args.seq2seq_ckpt:
        sd = torch.load(args.seq2seq_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd.get("model", sd), strict=False)
        print("[i] seq2seq cargado:", args.seq2seq_ckpt)

    app = build_app()
    app.state.model = model
    app.state.vocab = vocab
    app.state.clf = clf
    app.state.labels = labels
    app.state.seq2seq_trained = bool(args.seq2seq_ckpt)
    app.state.beam = args.beam
    app.state.max_len = args.max_len

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
