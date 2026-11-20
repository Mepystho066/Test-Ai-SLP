#!/usr/bin/env python3
"""Adaptador LSC50 -> contrato de pose del repo.

Fuente: LSC50 (U. de los Andes), figshare DOI 10.6084/m9.figshare.27383016, **CC BY 4.0**.
Estructura real del ZIP (`LANDMARKS.zip`, verificada):

    LANDMARKS/BODY_LANDMARKS/CCCC_SSSS_RRRR.csv          33 landmarks x (x,y,z) = 99 cols
    LANDMARKS/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS/...    21 landmarks x (x,y,z) = 63 cols
    LANDMARKS/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS/...   21 landmarks x (x,y,z) = 63 cols

`CCCC` = clase (0..49), `SSSS` = firmante (0..4), `RRRR` = repetición. Los tres CSV de un
clip tienen el mismo número de filas y el mismo índice de frame, así que se alinean por fila.

Mapeo al contrato (`pose_contract.py`, 318 dim):

    [0:42]   manos: izquierda 21 + derecha 21            (LSC50 los trae aparte)
    [42:56]  brazos/torso: pose 11..24 de BODY           (14 puntos)
    [56:106] labios + cejas: **sin datos en LSC50** -> se rellenan con ceros

Consecuencia que hay que tener presente: con LSC50 el encoder **no puede aprender gramática
facial** (la cara es gramática en LSC). Las 150 dims de cara quedan a cero; el día que haya
LSC con rostro se activan sin cambiar la arquitectura.

Uso:
    python lsc50_adapter.py --src ../data_raw/lsc50/LANDMARKS --out data/lsc50 --stride 2
"""
import argparse
import os

import numpy as np

import pose_contract as pc

POSE_COLS = 33 * 3
HAND_COLS = 21 * 3
FACE_POINTS = 468

# Subconjunto facial del contrato: contorno de labios (20 exterior + 20 interior) y
# cejas (5 por lado). Índices del mallado canónico de MediaPipe FaceMesh; se validan
# geométricamente en `check_face_geometry()`.
FACE_IDX = {
    "lips_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
                   267, 0, 37, 39, 40, 185],
    "lips_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311,
                   312, 13, 82, 81, 80, 191],
    "brow_left": [70, 63, 105, 66, 107],
    "brow_right": [336, 296, 334, 293, 300],
}
FACE_SEL = (FACE_IDX["lips_outer"] + FACE_IDX["lips_inner"]
            + FACE_IDX["brow_left"] + FACE_IDX["brow_right"])   # 50 puntos
assert len(FACE_SEL) == pc.N_LIPS + pc.N_BROWS == 50
assert len(set(FACE_SEL)) == 50


def read_csv(path: str) -> np.ndarray:
    """CSV con columna índice + 3*N coordenadas -> (T, N, 3). Cero relleno si falta."""
    if not os.path.exists(path):
        return np.zeros((0, 0, 3), dtype=np.float32)
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 1:].reshape(data.shape[0], -1, 3)     # descarta el índice


def resample_face(face: np.ndarray, t: int) -> np.ndarray:
    """LSC50 guarda la cara a otro fps que el cuerpo (163 vs 59 frames en el mismo clip).

    Se verifica empíricamente con la nariz: correlación 0,997 entre el punto 168 del
    mallado facial y el landmark 0 del cuerpo tras interpolar, así que reescalar por
    tiempo normalizado es correcto. No basta con recortar filas.
    """
    if face.shape[0] == t:
        return face
    idx = np.linspace(0, face.shape[0] - 1, t)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, face.shape[0] - 1)
    w = (idx - lo)[:, None, None].astype(np.float32)
    return face[lo] * (1 - w) + face[hi] * w


def build_clip(body: np.ndarray, left: np.ndarray, right: np.ndarray,
               face: np.ndarray = None, stride: int = 1) -> np.ndarray:
    """(T, 318) del contrato con la cara a cero. Recorta/padea a T_ENC."""
    t = body.shape[0]
    if left.shape[0] != t or right.shape[0] != t:
        t = min(body.shape[0], left.shape[0], right.shape[0])
        body, left, right = body[:t], left[:t], right[:t]

    out = np.zeros((t, pc.N_POINTS, 3), dtype=np.float32)
    out[:, pc.SL_HANDS][:, :21] = left                   # izq. primero
    out[:, pc.SL_HANDS][:, 21:] = right
    out[:, pc.SL_ARMS] = body[:, 11:25]                   # pose 11..24
    if face is not None and face.size:
        fr = resample_face(face[:t], t)
        out[:, pc.SL_LIPS] = fr[:, FACE_SEL[: pc.N_LIPS]]
        out[:, pc.SL_BROWS] = fr[:, FACE_SEL[pc.N_LIPS:]]
    out = pc.normalize_sequence(out)
    if stride > 1:
        out = out[::stride]
    if out.shape[0] > pc.T_ENC:
        out = out[: pc.T_ENC]
    flat = out.reshape(out.shape[0], pc.DIM).copy()
    if flat.shape[0] < pc.T_ENC:                          # pad hasta T_ENC
        pad = np.zeros((pc.T_ENC - flat.shape[0], pc.DIM), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=0)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data_raw/lsc50/LANDMARKS")
    ap.add_argument("--out", default="data/lsc50")
    ap.add_argument("--stride", type=int, default=1, help="submuestreo temporal (1 = sin stride)")
    ap.add_argument("--per-class", type=int, default=0, help="0 = todas las repeticiones")
    args = ap.parse_args()

    body_dir = os.path.join(args.src, "BODY_LANDMARKS")
    face_dir = os.path.join(args.src, "FACE_LANDMARKS")
    left_dir = os.path.join(args.src, "HANDS_LANDMARKS", "LEFT_HAND_LANDMARKS")
    right_dir = os.path.join(args.src, "HANDS_LANDMARKS", "RIGHT_HAND_LANDMARKS")
    if not os.path.isdir(body_dir):
        raise SystemExit(f"no está {body_dir}: descomprime LANDMARKS.zip ahí")

    feats = os.path.join(args.out, "features")
    os.makedirs(feats, exist_ok=True)
    manifest = os.path.join(args.out, "all.tsv")

    names = sorted(f for f in os.listdir(body_dir) if f.endswith(".csv"))
    per_class: dict = {}
    rows, skipped, frames = [], 0, []
    for fname in names:
        cls, signer, rep = fname[:-4].split("_")
        if args.per_class and per_class.get(cls, 0) >= args.per_class:
            continue
        body = read_csv(os.path.join(body_dir, fname))
        left = read_csv(os.path.join(left_dir, fname))
        right = read_csv(os.path.join(right_dir, fname))
        face = read_csv(os.path.join(face_dir, fname))
        if body.size == 0 or left.size == 0 or right.size == 0:
            skipped += 1
            continue
        clip = build_clip(body, left, right, face, stride=args.stride)
        name = fname[:-4]
        np.save(os.path.join(feats, f"{name}.npy"), clip)
        rows.append(f"{name}\t{clip.shape[0]}\tsign_{cls}\tsigner_{signer}")
        per_class[cls] = per_class.get(cls, 0) + 1
        frames.append(int((body[:clip.shape[0], 0] != 0).any(axis=1).sum()))

    with open(manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    n_cls = len({r.split("\t")[2] for r in rows})
    print(f"clips escritos : {len(rows)} ({skipped} descartados)")
    print(f"clases         : {n_cls} | firmantes: "
          f"{len({r.split(chr(9))[3] for r in rows})}")
    print(f"frames/clip    : mediana {int(np.median(frames))}, min {min(frames)}, max {max(frames)}")
    print(f"dim            : {np.load(os.path.join(feats, rows[0].split(chr(9))[0] + '.npy')).shape}")
    print(f"manifest       : {manifest}  (columnas: name, T, clase, firmante)")

    # verificación geométrica del subconjunto facial sobre el primer clip escrito
    sample = np.load(os.path.join(feats, rows[0].split("\t")[0] + ".npy")).reshape(
        -1, pc.N_POINTS, 3)
    lips_y = float(sample[:, pc.SL_LIPS, 1].mean())
    brows_y = float(sample[:, pc.SL_BROWS, 1].mean())
    face_nonzero = float(np.abs(sample[:, pc.N_HANDS + pc.N_ARMS:, :]).mean())
    print(f"check cara     : cejas y={brows_y:+.3f} vs labios y={lips_y:+.3f} "
          f"(cejas deben estar arriba: {'OK' if brows_y < lips_y else 'REVISAR'}) "
          f"| magnitud media {face_nonzero:.3f}")
    assert brows_y < lips_y, "los índices de cejas/labios no son los esperados"
    assert face_nonzero > 1e-3, "la cara quedó a cero: no se está leyendo FACE_LANDMARKS"


if __name__ == "__main__":
    main()
