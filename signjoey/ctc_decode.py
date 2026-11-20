"""CTC decoding en PyTorch puro — reemplazo del `tf.nn.ctc_beam_search_decoder`.

El fork usa TensorFlow en `model.py:331` **solo** para decodificar la salida de glosas
(rama `do_recognition`). Todo lo demás — beam search conjunto, greedy — ya es torch.
Este módulo quita esa última dependencia.

    log_probs: (T, B, V) log-probabilidades, blank = 0 por convención del repo.
    lengths:   (B,) longitud válida en frames.

Referencias: Hannun et al. 2014 (prefix beam search), Watanabe et al. 2017 (joint CTC).
"""
from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor

NEG_INF = float("-inf")


def _logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def log_softmax_frames(logits: Tensor) -> Tensor:
    """(B, T, V) -> (T, B, V) en log-espacio, como espera el decodificador."""
    return logits.log_softmax(dim=-1).permute(1, 0, 2)


def ctc_greedy_decode(log_probs: Tensor, lengths: Tensor, blank: int = 0) -> List[List[int]]:
    """Colapsa repeticiones y elimina blank. `lengths` recorta el padding."""
    best = log_probs.argmax(dim=-1)  # (T, B)
    out = []
    for b in range(best.size(1)):
        seq, prev = [], None
        for t in range(int(lengths[b])):
            v = int(best[t, b])
            if v != prev and v != blank:
                seq.append(v)
            prev = v
        out.append(seq)
    return out


def ctc_prefix_beam_search(
    log_probs: Tensor,
    lengths: Tensor,
    beam_width: int = 10,
    blank: int = 0,
    prune_top: int = 8,
) -> List[List[int]]:
    """Prefix beam search (Hannun et al.) en log-espacio, sin LM.

    Mantiene por prefijo las dos probabilidades que importan: terminar en blank y
    terminar en no-blank. Complejidad O(T · beam · prune_top) por muestra.
    """
    assert log_probs.dim() == 3, log_probs.shape
    n_batch = log_probs.size(1)
    results: List[List[int]] = []

    for b in range(n_batch):
        beams = {(): (0.0, NEG_INF)}          # prefijo -> (log p_blank, log p_nonblank)
        for t in range(int(lengths[b])):
            lp = log_probs[t, b]
            top = torch.topk(lp, min(prune_top + 1, lp.numel())).indices.tolist()
            new: dict = {}
            for prefix, (p_b, p_nb) in beams.items():
                # 1) no emitir (blank): el prefijo se mantiene
                acc_b, acc_nb = new.get(prefix, (NEG_INF, NEG_INF))
                acc_b = _logaddexp(acc_b, _logaddexp(p_b, p_nb) + float(lp[blank]))
                new[prefix] = (acc_b, acc_nb)
                # 2) emitir un token sobre el prefijo
                for v in top:
                    if v == blank:
                        continue
                    pv = float(lp[v])
                    if prefix and prefix[-1] == v:
                        # repetición: se colapsa sobre el mismo prefijo (solo desde no-blank)
                        b0, nb0 = new[prefix]
                        new[prefix] = (b0, _logaddexp(nb0, p_nb + pv))
                    else:
                        ext = prefix + (v,)
                        b0, nb0 = new.get(ext, (NEG_INF, NEG_INF))
                        new[ext] = (b0, _logaddexp(nb0, _logaddexp(p_b, p_nb) + pv))
            beams = dict(sorted(
                new.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]), reverse=True
            )[:beam_width])
        best = max(beams.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]))[0]
        results.append(list(best))
    return results


def ctc_score(log_probs: Tensor, seq: List[int], length: int, blank: int = 0) -> float:
    """log P(seq | frames) por la recursión forward de CTC, para re-puntuar hipótesis.

    `seq` es la secuencia **colapsada** (sin blanks). Devuelve -inf si no cabe.
    """
    ext = [blank]
    for s in seq:
        ext += [s, blank]
    if length < len(ext) - 1:
        return NEG_INF

    alpha = [NEG_INF] * len(ext)
    alpha[0] = float(log_probs[0, 0, blank])
    if len(ext) > 1:
        alpha[1] = float(log_probs[0, 0, ext[1]])

    for t in range(1, length):
        new = [NEG_INF] * len(ext)
        for s in range(len(ext)):
            acc = alpha[s]
            if s - 1 >= 0:
                acc = _logaddexp(acc, alpha[s - 1])
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2]:
                acc = _logaddexp(acc, alpha[s - 2])
            new[s] = acc + float(log_probs[t, 0, ext[s]])
        alpha = new

    out = alpha[-1]
    if len(ext) > 1:
        out = _logaddexp(out, alpha[-2])
    return out


# --- compatibilidad con la firma que usa el fork (batch) -----------------------

def ctc_beam_search_batch(
    logits: Tensor, lengths: Tensor, beam_width: int = 10, blank: int = 0
) -> List[List[int]]:
    """Atajo: recibe logits (B, T, V) + longitudes y devuelve secuencias colapsadas."""
    log_probs = log_softmax_frames(logits)
    if beam_width < 2:
        return ctc_greedy_decode(log_probs, lengths, blank=blank)
    return ctc_prefix_beam_search(log_probs, lengths, beam_width=beam_width, blank=blank)
