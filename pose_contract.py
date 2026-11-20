"""Contrato del encoder de pose (v0) — fuente única de verdad.

Acordado 2026-10-30 con el equipo (Hito 0). Produce exactamente lo que espera el
YAML de SignJoey del fork Tan et al. (`configs/mlslt_smoke.yaml`):

    sgn_dim = 318   features CRUDAS por frame (106 landmarks x x,y,z)
    hidden  = 256   SpatialEmbeddings(input_size=318, embedding_dim=256)
    T_max   = 256 frames @25 fps, stride 2 -> T_enc = 128 pasos

Verificado contra el repo (signjoey/model.py:453 "sgn_dim: feature dimension of the
sign frame representation", :464 proyección, :476 "emb_size must be hidden_size").

Este módulo no importa torch a propósito: solo numpy, para que lo puedan usar el
generador sintético, el día que exista el extractor real de MediaPipe, y los tests.
"""
import numpy as np

# --- MediaPipe Holistic: 543 puntos = 33 pose + 21x2 manos + 468 cara ---
POSE, HAND, FACE = 33, 21, 468
assert POSE + 2 * HAND + FACE == 543

# --- Subconjunto de la nota: manos + brazos/torso + labios + cejas ---
N_HANDS = 2 * HAND   # 42
N_ARMS = 14          # pose 11..24: hombros, codos, muñecas, dedos-pose, caderas
N_LIPS = 40          # 20 exterior + 20 interior
N_BROWS = 10         # 5 por lado
N_POINTS = N_HANDS + N_ARMS + N_LIPS + N_BROWS   # 106

DIM = N_POINTS * 3      # 318
T_MAX = 256             # frames @25 fps (10,24 s)
STRIDE = 2
T_ENC = T_MAX // STRIDE  # 128
MIN_SHOULDER_WIDTH = 1e-3

# layout plano (en puntos, no en dims)
SL_HANDS = slice(0, N_HANDS)
SL_ARMS = slice(N_HANDS, N_HANDS + N_ARMS)
SL_LIPS = slice(N_HANDS + N_ARMS, N_HANDS + N_ARMS + N_LIPS)
SL_BROWS = slice(N_HANDS + N_ARMS + N_LIPS, N_POINTS)

# pose 11 = hombro izquierdo, pose 12 = hombro derecho -> dentro del bloque de brazos
SH_L = N_HANDS + 0
SH_R = N_HANDS + 1

# Holistic completo, por si se ablaciona
DIM_FULL = 543 * 3      # 1629


def normalize_sequence(points: np.ndarray) -> np.ndarray:
    """Normaliza una secuencia (T, N_POINTS, 3) de coords normalizadas de MediaPipe.

    Traslada al punto medio de los hombros y escala por la anchura de hombros, para
    que la señal no dependa de la distancia a la cámara ni de la posición en cuadro.

    Ojo: los hombros caen en SL_ARMS[0] y SL_ARMS[1] del vector plano, NO en 0 y 1.
    Indexarlos mal produce features con |x| hasta 20x lo esperado (visto en pruebas).
    """
    pts = np.asarray(points, dtype=np.float32)
    center = (pts[:, SH_L] + pts[:, SH_R]) / 2.0                     # (T,3)
    width = np.linalg.norm(pts[:, SH_L] - pts[:, SH_R], axis=-1)     # (T,)
    width = np.maximum(width, MIN_SHOULDER_WIDTH)[:, None, None]
    return (pts - center[:, None, :]) / width


def synthetic_sequence(rng: np.random.Generator, t_len: int = T_MAX) -> np.ndarray:
    """Landmarks sintéticos ya normalizados: (T_ENC, DIM) float32.

    Prueba el cableado (shapes, máscaras, CTC, beam search). NO prueba transferencia:
    no hay coarticulación, ni gramática facial, ni variación de firmante.
    """
    pts = rng.normal(0, 0.03, size=(t_len, N_POINTS, 3)).astype(np.float32)
    phase = np.linspace(0, 4 * np.pi, t_len)[:, None]
    # manos: oscilación de ~0.15 anchuras de hombro
    pts[:, SL_HANDS, 0] += 0.15 * np.sin(phase)
    pts[:, SL_HANDS, 1] += 0.10 * np.cos(phase)
    # cara: casi estática (solo micromovimiento)
    pts[:, SL_LIPS] *= 0.2
    pts[:, SL_BROWS] *= 0.2
    # hombros fijos: si son aleatorios la anchura puede ser ~0 y la normalización explota
    pts[:, SH_L] = np.array([-0.5, 0.0, 0.0])
    pts[:, SH_R] = np.array([0.5, 0.0, 0.0])
    pts = normalize_sequence(pts)
    return pts[:, ::STRIDE].reshape(-1, DIM).copy()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = synthetic_sequence(rng)
    print("sgn_dim  :", DIM, f"({N_POINTS} puntos x 3)")
    print("T_MAX/T_ENC:", T_MAX, "/", T_ENC, f"(stride {STRIDE})")
    print("tensor   :", x.shape, x.dtype, "| max |x| =", round(float(np.abs(x).max()), 3))
    print("embeddings 318->256:", DIM * 256, "params")
    assert x.shape == (T_ENC, DIM) and np.isfinite(x).all()
    assert not np.allclose(x, 0), "una secuencia normalizada no puede ser todo ceros"
    print("OK contrato v0")
