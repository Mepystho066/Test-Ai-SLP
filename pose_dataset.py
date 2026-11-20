"""Dataset de pose nativo de PyTorch (sin torchtext).

El fork carga datos con `torchtext.data.Dataset`/`Field` (API eliminada en torchtext
>=0.9) y con un loader hardcodeado a CSL-Daily (`.npy` con `original_info`/`label_char`
+ features SlowFast + h5py). Aquí va la capa de datos que sí corre hoy:

    manifest.tsv  ->  <split>/<name>.npy  (T, 318)  +  texto objetivo (una línea)

Formato del manifest (TSV, sin cabecera):
    <name>\\t<seq_len>\\t<texto de la frase>

`seq_len` es la longitud real en frames (antes del stride); el .npy puede venir ya
en T_ENC. El texto se tokeniza a nivel de palabra; el vocabulario se construye de los
propios datos (Hito 1: reemplazar por SentencePiece mT5 recortado a 50-80k).

Uso:
    from pose_dataset import PoseManifest, collate
    ds = PoseManifest("data/train.tsv", txt_vocab)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from signjoey.batch import Batch

BOS, EOS, UNK, PAD = 2, 3, 0, 1   # convención de signjoey.vocabulary.TextVocabulary
BLANK = 0                          # índice que usa nn.CTCLoss como blank


def encode_text(text: str, vocab, bos: bool = True, eos: bool = True) -> list:
    """Tokeniza a nivel de palabra y mapea a ids con el `stoi` del vocabulario.

    `TextVocabulary` del fork no expone `encode()` (solo `array_to_sentence`), y su
    `stoi` es un defaultdict con UNK=0, así que las palabras fuera del vocab caen
    solas en UNK.
    """
    # TrimmedVocab (SentencePiece) expone encode_ids; TextVocabulary del fork solo stoi.
    if hasattr(vocab, "encode_ids"):
        return vocab.encode_ids(text, bos=bos, eos=eos)
    ids = [vocab.stoi[t.lower()] for t in text.strip().split()]
    if bos:
        ids.insert(0, BOS)
    if eos:
        ids.append(EOS)
    return ids


@dataclass
class Sample:
    name: str
    sgn: torch.Tensor      # (T_enc, DIM)
    gls: torch.Tensor      # (L,) ids de glosa
    txt: torch.Tensor      # (L,) ids de texto, ya con BOS/EOS
    lid: int               # id de lengua de origen (gsg/csl/lsc), CTC para el LID head


# Mapeo gsg/csl/lsc -> id LID (token-level, CTC sobre la capa inicial del encoder).
# Los ids empiezan en 1: el 0 es el blank de nn.CTCLoss, así que mapear la primera
# lengua a 0 haría que el target fuera todo blank y la pérdida LID quedara en 0.
LID_IDS = {"gsg": 1, "csl": 2, "lsc": 3}

# Por defecto gsg para el eval del Hito 1 (PHOENIX). Para entrenamiento
# multi-lengua se sobreescribe con la columna 4era del manifest.
LID_DEFAULT = LID_IDS["gsg"]


class PoseManifest(Dataset):
    """Lee un manifest TSV y devuelve tensores ya recortados a T_MAX.

    Formato:
        <name>  <seq_len>  <texto>     [lid]   (3 o 4 columnas)
    `seq_len` es la longitud real en frames (antes del stride); el .npy puede venir
    ya en T_ENC. `texto` se tokeniza a nivel de palabra; `lid` es la id de la
    lengua de origen (gsg=1, csl=2, lsc=3; 0 queda reservado como blank del CTC). El eval del Hito 1 (PHOENIX) se queda
    en el subconjunto de PHOENIX (gsg). Para entrenamiento multi-lengua se añade
    la columna 4ta con la id LID.
    """

    def __init__(self, manifest: str, txt_vocab, gls_vocab=None,
                 feat_dir: str = "data/features", t_max: int = 128):
        self.items: List[tuple] = []
        self.feat_dir = feat_dir
        self.t_max = t_max
        self.txt_vocab = txt_vocab
        self.gls_vocab = gls_vocab
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    raise ValueError(f"manifest mal formado (<3 columnas): {line!r}")
                name, seq_len, text = parts[0], int(parts[1]), parts[2]
                lid = int(parts[3]) if len(parts) >= 4 else LID_DEFAULT
                if lid not in LID_IDS.values():
                    raise ValueError(f"{name}: lid={lid} fuera de rango")
                self.items.append((name, seq_len, text, lid))
        if not self.items:
            raise ValueError(f"manifest vacío: {manifest}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        name, seq_len, text, lid = self.items[idx]
        feats = np.load(f"{self.feat_dir}/{name}.npy").astype(np.float32)
        feats = feats[: self.t_max]                     # recorte al contrato
        if feats.shape[1] != 318:
            raise ValueError(f"{name}: se esperaba DIM=318, llegó {feats.shape}")
        txt_ids = encode_text(text, self.txt_vocab)
        gls_ids = encode_text(text, self.gls_vocab, bos=False, eos=False) \
            if self.gls_vocab is not None else []
        return Sample(
            name=name,
            sgn=torch.from_numpy(feats),
            gls=torch.tensor(gls_ids, dtype=torch.long),
            txt=torch.tensor(txt_ids, dtype=torch.long),
            lid=lid,
        )


def _pad(seqs: Sequence[torch.Tensor], pad_value: int):
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    out = torch.full((len(seqs), int(lengths.max())), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, lengths


def collate(samples: Sequence[Sample], txt_pad_index: int = PAD,
            gls_pad_index: int = 2) -> Batch:
    """Construye el `Batch` de signjoey a partir de muestras de tamaño variable."""
    from types import SimpleNamespace

    lengths = [s.sgn.shape[0] for s in samples]
    t_max = max(lengths)
    sgn = torch.zeros(len(samples), t_max, samples[0].sgn.shape[1])
    for i, s in enumerate(samples):
        sgn[i, : s.sgn.shape[0]] = s.sgn
    sgn_lengths = torch.tensor(lengths, dtype=torch.long)

    txt, txt_lengths = _pad([s.txt for s in samples], txt_pad_index)
    gls, gls_lengths = _pad([s.gls for s in samples], gls_pad_index)

    # LIDtok (paper §3.2): misma longitud que el texto hablado y cada posición con
    # el id de lengua de origen (0 = blank del CTC). `Batch.__init__` lo desempaqueta
    # como `(lid, lid_lengths)` (batch.py:121), así que aquí va una tupla: pasar un
    # tensor 1-D suelto desemboca en "too many values to unpack".
    lid = torch.zeros_like(txt)
    for i, s in enumerate(samples):
        lid[i, : int(txt_lengths[i])] = s.lid

    tb = SimpleNamespace(
        sequence=[s.name for s in samples],
        signer=["synthetic"] * len(samples),
        sgn=(sgn, sgn_lengths),
        txt=(txt, txt_lengths),
        gls=(gls, gls_lengths),
        lid=(lid, txt_lengths.clone()),
    )
    batch = Batch(is_train=True, torch_batch=tb, txt_pad_index=txt_pad_index,
                  sgn_dim=sgn.shape[-1], use_cuda=False)
    # Batch infiere la máscara comparando frames con el vector cero: un frame
    # legítimamente nulo (o con x==0 en el primer landmark) se perdería. La máscara
    # correcta sale de las longitudes reales.
    # dtype bool: transformer_layers.py:73 hace `~mask` y el encoder espera
    # (B, 1, T) booleano, igual que el mask original de Batch.
    batch.sgn_mask = (
        torch.arange(t_max).unsqueeze(0) < sgn_lengths.unsqueeze(1)
    ).unsqueeze(1)
    return batch
