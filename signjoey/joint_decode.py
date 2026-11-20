"""Decodificación conjunta CTC/atención en PyTorch puro (reemplazo limpio).

El fork ya trae `search_jointdecoding_new.py` (torch, sin TF), pero su fusión con CTC
tiene tres problemas que este módulo corrige:

1. **Recalcula el CTC en cada paso**: `ctc_layer(encoder_output)` dentro del bucle, T veces.
   Aquí se calcula una sola vez y se reutiliza.
2. **No usa una puntuación CTC de secuencia**: suma `log p(token)` en *todos* los frames
   (`topk_ctc_scores ... .sum(dim=1)`) y normaliza con z-score. Eso no es la probabilidad
   del prefijo y en la práctica es ruido; aquí se usa la recursión forward de CTC
   (`ctc_decode.ctc_score`), que sí es log P(seq | frames).
3. **Mezcla dos criterios**: el beam avanza por atención (`topk_log_probs`) pero las
   hipótesis se ordenan por `combined_scores`. Aquí el beam avanza por atención con
   penalización de longitud y la fusión CTC se aplica **al final**, re-puntuando las
   hipótesis completas (joint decoding de Watanabe et al.). Es más barato y reproducible.

API:
    joint_decode(decoder, embed, encoder_output, encoder_hidden, src_mask,
                 bos, eos, pad, max_output_length, ctc_layer=None,
                 ctc_weight=0.3, beam=4, len_penalty=0.0)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from signjoey.ctc_decode import ctc_score, log_softmax_frames

NEG_INF = float("-inf")


def _length_penalty(step: int, alpha: float) -> float:
    return ((5.0 + step) / 6.0) ** alpha if alpha > 0 else 1.0


@torch.no_grad()
def joint_decode(
    decoder,
    embed,
    encoder_output: Tensor,      # (B, T, H)
    encoder_hidden: Tensor,
    src_mask: Tensor,            # (B, 1, T) bool
    bos_index: int,
    eos_index: int,
    pad_index: int,
    max_output_length: int,
    ctc_layer=None,
    ctc_weight: float = 0.3,
    beam: int = 4,
    len_penalty: float = 0.0,
) -> Tuple[List[List[int]], List[float]]:
    """Devuelve hipótesis (sin BOS, sin EOS) y su puntuación final por muestra."""
    assert beam >= 1
    batch_size = encoder_output.size(0)
    device = encoder_output.device

    ctc_log_probs = None
    if ctc_layer is not None and ctc_weight > 0:
        ctc_log_probs = log_softmax_frames(ctc_layer(encoder_output))   # (T, B, V)

    all_hyps: List[List[int]] = []
    all_scores: List[float] = []

    for b in range(batch_size):
        enc = encoder_output[b: b + 1].expand(beam, -1, -1).contiguous()
        hid = None if encoder_hidden is None else encoder_hidden[:, b: b + 1].expand(
            -1, beam, -1).contiguous()
        mask = src_mask[b: b + 1].expand(beam, -1, -1).contiguous()
        trg_mask = mask.new_ones([1, 1, 1])

        # prefijo -> log-prob acumulada (atención)
        alive = {(): 0.0}
        finished: List[Tuple[Tuple[int, ...], float]] = []

        for step in range(max_output_length):
            prefixes = list(alive.keys())
            if not prefixes:
                break
            n = len(prefixes)
            trg = torch.full((n, step + 1), pad_index, dtype=torch.long, device=device)
            for i, p in enumerate(prefixes):
                trg[i, 0] = bos_index
                if p:
                    trg[i, 1:len(p) + 1] = torch.tensor(p, device=device)
            logits, _, _, _ = decoder(
                encoder_output=enc[:n], encoder_hidden=None if hid is None else hid[:, :n],
                src_mask=mask[:n], trg_embed=embed(trg), hidden=None,
                prev_att_vector=None, unroll_steps=1, trg_mask=trg_mask,
            )
            logp = F.log_softmax(logits[:, -1], dim=-1)     # (n, V)

            cand: dict = {}
            for i, p in enumerate(prefixes):
                base = alive[p]
                if len(p) + 1 >= max_output_length:
                    finished.append((p, base))
                    continue
                lp = logp[i]
                k = min(beam + 1, lp.numel())
                top_lp, top_v = lp.topk(k)   # torch.topk devuelve (values, indices)
                pen = _length_penalty(step + 1, len_penalty)
                for v, s in zip(top_v.tolist(), top_lp.tolist()):
                    if v == pad_index:
                        continue
                    score = base + s
                    if v == eos_index:
                        finished.append((p, score))
                        continue
                    np_ = p + (v,)
                    prev = cand.get(np_, NEG_INF)
                    cand[np_] = score if prev == NEG_INF else max(prev, score)
            # recorta al ancho del beam, normalizando por longitud para comparar prefijos
            alive = dict(sorted(
                cand.items(),
                key=lambda kv: kv[1] / _length_penalty(len(kv[0]), len_penalty),
                reverse=True,
            )[:beam])
            if not alive and finished:
                break

        if not alive and not finished:
            all_hyps.append([])
            all_scores.append(NEG_INF)
            continue

        # re-puntuación final: atención + peso * CTC de secuencia
        pool = finished + [(p, s) for p, s in alive.items()]
        scored = []
        for p, s in pool:
            total = s / _length_penalty(max(len(p), 1), len_penalty)
            if ctc_log_probs is not None and p:
                cs = ctc_score(ctc_log_probs[:, b: b + 1], list(p),
                               int(src_mask[b, 0].sum()))
                total = total + ctc_weight * cs
            scored.append((total, p))
        scored.sort(reverse=True)
        best_score, best = scored[0]
        all_hyps.append(list(best))
        all_scores.append(float(best_score))

    return all_hyps, all_scores
