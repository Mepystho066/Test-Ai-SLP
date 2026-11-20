"""Vocabulario recortado de mT5 (64k) listo para la cabeza CTC de SignJoey.

Artefactos (de @default, en ../vocab/):
    mt5_trimmed_64000.model     SentencePiece podado (encode re-segmenta con lo retenido)
    mt5_trimmed_64000.map.json  {'piece_to_idx': pieza -> idx, 'id_modelo_to_idx': id_sp -> idx}

Convención del fork: 0=UNK, 1=PAD, 2=BOS, 3=EOS; 0 es también el blank del CTC.
Las piezas ocupan 4..64003 -> **el tamaño de la cabeza es 64004**, no el
`vocab_size_total: 64003` que declara el JSON (si se usa ese número, la última
pieza, `▁Rij`, queda fuera del embedding).

Uso:
    from pose_vocab import TrimmedVocab
    v = TrimmedVocab("vocab/mt5_trimmed_64000.model", "vocab/mt5_trimmed_64000.map.json")
    ids = v.encode("hola, ¿cómo estás?")          # -> [4..64003]
    v.decode(ids)                                  # -> texto
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Iterable, List

UNK, PAD, BOS, EOS = 0, 1, 2, 3
OFFSET = 4


class TrimmedVocab:
    def __init__(self, model_path: str, map_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        meta = json.load(open(map_path, encoding="utf-8"))
        # JSON: claves str -> convertir a int
        self.id_sp_to_idx = {int(k): int(v) for k, v in meta["id_modelo_to_idx"].items()}
        self.piece_to_idx = {k: int(v) for k, v in meta["piece_to_idx"].items()}
        self.idx_to_piece = {v: k for k, v in self.piece_to_idx.items()}
        self.vocab_size = max(self.idx_to_piece) + 1          # 64004
        self.specials = {UNK: "<unk>", PAD: "<pad>", BOS: "<s>", EOS: "</s>"}
        # build_model() consulta txt_vocab.stoi[PAD_TOKEN] etc.; con SentencePiece
        # esas piezas no existen en el .model, así que se resuelven aquí.
        self.stoi = {"<unk>": UNK, "<pad>": PAD, "<s>": BOS, "</s>": EOS}

    # --- API que espera pose_dataset -----------------------------------------
    def encode_ids(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:
        ids = [self.id_sp_to_idx.get(i, UNK) for i in self.sp.encode(text, out_type=int)]
        if bos:
            ids.insert(0, BOS)
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        pieces = []
        for i in ids:
            i = int(i)
            if i in (BOS, EOS, PAD):
                continue
            pieces.append(self.specials.get(i) or self.idx_to_piece.get(i, "<unk>"))
        return "".join(pieces).replace("▁", " ").strip()

    # compatibilidad con el patrón `vocab.stoi` del fork
    def __len__(self) -> int:
        return self.vocab_size

    def unk_rate(self, texts: Iterable[str]) -> float:
        """Fracción de piezas que caen en UNK: el número que hay que vigilar al recortar."""
        total = unk = 0
        for t in texts:
            ids = self.sp.encode(t, out_type=int)
            total += len(ids)
            unk += sum(1 for i in ids if i not in self.id_sp_to_idx)
        return unk / max(total, 1)


if __name__ == "__main__":
    import os
    import sys

    mp = sys.argv[1] if len(sys.argv) > 1 else "../vocab/mt5_trimmed_64000.model"
    jp = sys.argv[2] if len(sys.argv) > 2 else "../vocab/mt5_trimmed_64000.map.json"
    if not os.path.exists(mp):
        raise SystemExit(f"no está el modelo: {mp} (¿corriste desde la raíz del repo?)")
    v = TrimmedVocab(mp, jp)
    print(f"tamaño de la cabeza CTC: {v.vocab_size} (piezas 4..{v.vocab_size - 1})")

    samples = {
        "es": ["hola, ¿cómo estás?", "necesito un intérprete para la cita médica",
               "mañana a las ocho en la casa de mi hermana", "señal, pequeño, reloj"],
        "de": ["Guten Morgen, morgen regnet es stark", "Wie ist das Wetter am Wochenende?"],
        "en": ["the weather forecast for tomorrow is rain"],
        "zh": ["明天会下雨吗", "请帮我翻译这句话"],
    }
    for lang, texts in samples.items():
        joined = " ".join(texts)
        print(f"  {lang}: unk={v.unk_rate(texts):.3f} piezas={len(v.sp.encode(joined))}")
    demo = "necesito un intérprete"
    print("  demo:", demo, "->", v.decode(v.encode_ids(demo, bos=False, eos=False)))
