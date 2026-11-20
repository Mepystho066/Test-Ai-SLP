# visual-mlslt — notas de desarrollo (Hito 0)

Fork de `Claire874/Gloss-free-MLSLT` (Tan et al., ACL 2025) con encoder de **pose
MediaPipe Holistic** en lugar de features SlowFast RGB.

## Arrancar en frío

```bash
uv venv .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv/bin/python numpy pyyaml

.venv/bin/python pose_contract.py            # contrato: (128, 318) + normalización
.venv/bin/python test_pose_pipeline.py       # regresiones del contrato (máscara, normalización)
.venv/bin/python smoke_test.py               # forward/backward de las 3 pérdidas
.venv/bin/python train_smoke.py --steps 40   # 40 pasos de Adam sobre datos sintéticos
.venv/bin/python lid_smoke.py                # cabeza Sign2LID (Batch montado a mano)
.venv/bin/python test_lid_pipeline.py        # LID end-to-end desde el loader de pose
.venv/bin/python decode_smoke.py             # decodificación CTC + beam sin TensorFlow
```

Salidas esperadas: `[PASS] smoke test superado` y `[PASS] entrenamiento mínimo sobre
contrato de pose`, con la pérdida bajando (≈470 → ≈180 en 40 pasos, CPU).

## Contrato del encoder (v0) — `pose_contract.py`

| Campo | Valor |
|---|---|
| Landmarks | MediaPipe Holistic 543 (33 pose + 21×2 manos + 468 cara) |
| Subconjunto | manos 42 + brazos/torso 14 (pose 11–24) + labios 40 + cejas 10 = 106 puntos |
| `sgn_dim` | **318** (crudo; `model.py:453` lo documenta como dim de features, no de hidden) |
| `hidden_size` | **256** (= `embedding_dim`, exigido en `model.py:476`) |
| `T_max` / `T_enc` | **256** frames @25 fps, stride 2 → **128** |
| Normalización | centrar en hombros y escalar por anchura de hombros (`MIN_SHOULDER_WIDTH`) |

Los hombros (pose 11/12) caen en `N_HANDS+0` y `N_HANDS+1` del vector plano, **no** en
0 y 1: mal indexado da features con |x| hasta 20× y el modelo entrena basura.

## Datos — `pose_dataset.py`

- `manifest.tsv`: `<name>\t<seq_len>\t<texto>`; features en `data/.../features/<name>.npy`
  con shape `(T_enc, 318)`.
- `collate()` arma el `Batch` de signjoey y **reemplaza la máscara**:
  `Batch.__init__` infiere `sgn_mask` comparando frames con el vector cero
  (`(self.sgn != torch.zeros(sgn_dim))[..., 0]`), así que un frame legítimamente nulo se
  descarta como padding. La máscara buena sale de `sgn_lengths` y debe ser `bool`
  (`transformer_layers.py:73` hace `~mask`).
- `TextVocabulary` no tiene `encode()`; `encode_text()` usa su `stoi` (defaultdict → UNK=0).

## Hito 1: decodificación sin TensorFlow

TensorFlow **ya no se usa**. Corrección importante: el beam search conjunto
(`search_jointdecoding_new.py`) *ya* era torch; TF aparecía solo en `model.py:331` para
decodificar el CTC de **glosas**. Se sustituyó por `signjoey/ctc_decode.py`.

| Pieza | Archivo | Qué hace |
|---|---|---|
| CTC greedy + prefix beam | `signjoey/ctc_decode.py` | colapso, blank, re-puntuación por recursión forward (`ctc_score`) |
| Decode conjunto | `signjoey/joint_decode.py` | beam de atención + rescoring CTC al final (Watanabe et al.) |
| Prueba | `decode_smoke.py` | asserta que `tensorflow` nunca entra en `sys.modules` |

Bugs del fork que aparecieron al cablear esto:

1. **`encoders.py`: `specific_layer` era `None` con `num_layers < 4`.** El índice
   `len - ceil(len/2) + 1` con 2 capas da 2 y nunca se alcanza; `layer_norm(None)` reventaba.
   Corregido con `min(target, len-1)`; con 6 capas el índice no cambia (checkpoint intacto).
2. **`search_jointdecoding_new.py` recalcula el CTC en cada paso** y su “puntuación CTC”
   suma `log p(token)` en todos los frames (no es P(seq|frames)); además ordena hipótesis con
   `combined_scores` pero avanza el beam con atención. Se dejó intacto (no romper al equipo)
   y `joint_decode.py` ofrece la versión correcta para comparar.
3. **`torch.topk` devuelve (values, indices)**: usarlo como `top_v, top_lp = ...` mete floats
   como ids de token y el `embedding` revienta con `IndexError`. Trampa fácil de repetir.
4. **`collate` de `pose_dataset.py` pasaba `lid` como tensor 1-D suelto**, pero
   `Batch.__init__` (batch.py:121) lo desempaqueta como `(lid, lid_lengths)` →
   `ValueError: too many values to unpack`. Ahora `collate` arma el LIDtok con la
   longitud del texto hablado y sus `txt_lengths`.
5. **`LID_IDS` mapeaba `gsg=0`, que es el blank de `nn.CTCLoss`**: el target quedaba
   entero en blank y la pérdida LID salía 0 sin avisar. Ahora `gsg=1, csl=2, lsc=3`
   (coherente con `lid_vocab_size: 4` = 3 lenguas + blank).
6. **`test_lid_pipeline.py`** cubre lo que `lid_smoke.py` no: el `Batch` sale del loader
   real (`PoseManifest` + `collate`), no montado a mano.

## Vocabulario mT5 recortado (`pose_vocab.py`)

- `TrimmedVocab` carga el `.model` podado + `.map.json`, expone `encode_ids()` / `decode()`.
- **La cabeza CTC tiene 64004 clases**, no 64003: las piezas van de 4 a 64003 y el
  `vocab_size_total` del JSON deja fuera `▁Rij`.
- `tied_softmax: true` en el config: con 64004 clases baja de 55,1 M a 38,7 M params
  (2+2 capas); con 6+6 queda en 50,3 M.
- UNK medido: es 0,000 · de 0,000 · en 0,000 · zh 0,062 (coincide con @default).

## UI de cámara (`webcam_view.html`)

Widget autocontenido que captura la webcam, corre MediaPipe Tasks Vision (pose + manos
+ cara, equivalente al Holistic de 543 puntos) en el navegador, extrae los 106 landmarks
del contrato, normaliza por hombros y dibuja el **feature map 128×318** que consume el
modelo. El loop cámara → features ya está cableado; la traducción sale cuando haya
checkpoint. No hay backend: exporta `features.json` para alimentar Python.

- CDN: `@mediapipe/tasks-vision@1.2.0/vision_bundle.mjs` + wasm (jsdelivr); modelos
  `pose_landmarker_lite`, `hand_landmarker`, `face_landmarker` (storage.googleapis.com).
- FaceLandmarker devuelve **478** puntos (iris en 468..477); se usan los primeros 468,
  equivalentes a Holistic.
- **Índices FaceMesh usados como fuente de verdad** (el extractor Python debe replicarlos):
  - labios exteriores (20): `61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146`
  - labios interiores (20): `78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95`
  - cejas (10): `70,63,105,66,107,336,296,334,293,300`
- Orden del vector: manos izq(21) + der(21) → pose 11..24 → labios ext+int → cejas.
  El overlay se dibuja en espejo por UX, pero **los features NO se espejan**.
- Si `getUserMedia` no está disponible en el preview de Hermes, abrir el HTML en un
  navegador (o `python -m http.server` y `localhost:8000/webcam_view.html`).

## Backend de inferencia (`infer_server.py`)

FastAPI que carga un SignJoey (2+2 capas, tied softmax) sobre el vocab mT5 recortado y
expone `POST /predict` (features 128×318 → texto) y sirve `webcam_view.html` en `/`.

```bash
.venv/bin/python infer_server.py --port 8000   # luego http://localhost:8000
```

- Decodifica con `joint_decode` (beam atención + rescoring CTC), `ctc_weight=0.3`,
  `beam=4`, `max_len=40`. ~1.3 s por frase en CPU.
- **Pesos aleatorios**: el texto sale basura multilingüe real (piezas SentencePiece),
  no una traducción. Sirve para cerrar el loop cámara → features → texto.
- Trampa: `PredictBody` debe estar a **nivel de módulo**; definirlo dentro de
  `build_app()` rompe Pydantic v2 (`ForwardRef('PredictBody') is not fully defined`) y
  devuelve 422 "missing body" en cada POST.

## Hito A: clasificación de señas aisladas (LSC50)

```bash
# 1. descomprimir LANDMARKS.zip (figshare, CC BY 4.0) en ../data_raw/lsc50
# 2. adaptar al contrato
.venv/bin/python lsc50_adapter.py --per-class 16      # 800 clips, 50 clases
# 3. entrenar el encoder (clasificación, no BLEU)
.venv/bin/python train.py --manifest data/lsc50/all.tsv --feat-dir data/lsc50/features \
    --epochs 30 --batch 16 --layers 6 --val-frac 0.2 --out models/lsc50_enc
# 4. comprobar que el encoder entra en el seq2seq
.venv/bin/python test_transfer.py
```

- `lsc50_adapter.py` lee BODY/HANDS/FACE, reescala la cara (va a otro fps), normaliza con el
  contrato y deja `data/lsc50/all.tsv` con `name, T, clase, firmante`.
- `train.py` hace split **por firmante** cuando el manifest trae la 4ª columna, y reporta
  accuracy + F1 macro. Métrica de clasificación: aquí BLEU no aplica.
- `test_transfer.py` comprueba que `encoder_only.pt` (101 tensores) entra sin desajustes en
  el seq2seq del fork.

### Resultados reales (LSC50, split por firmante, CPU)

| Config | Mejor val acc | F1 macro | Notas |
|---|---|---|---|
| 800 clips, 6 capas, 30 épocas | **0,275** | 0,238 | azar 0,02; mejor época 12 |
| 1 000 clips, 2 capas, 12 épocas | 0,215 | 0,180 | mismas condiciones, menos profundidad |

Train loss baja a 0,39 mientras val loss sube a 5,13: **sobreajuste claro**. Con 1 000 clips y
5 firmantes el dataset es el techo, no el modelo. `encoder_only.pt` (6 capas, 37 MB) se
transfiere al seq2seq sin desajustes (`test_transfer.py`).

## Servidor de inferencia (`infer_server.py`)

Estado real del endpoint:

| Ruta | Qué devuelve |
|---|---|
| `GET /` | `webcam_view.html` (MediaPipe en el navegador, manda 106 landmarks/318 dim) |
| `POST /predict` | `sign` + `confidence` + `top3` (clasificador LSC50 **entrenado**) |
| `GET /health` | `trained: {classifier_lsc50: true, translation: false}` |

- El servidor carga `models/lsc50_enc/best.pt` (6 capas, 50 señas). Medido sobre 25 muestras
  de validación (firmante no visto): **7/25 = 0,28 top-1**, ~8 ms por petición.
- La **traducción seña→frase sigue sin checkpoint**: el seq2seq se instancia aleatorio y
  `/predict` ya no lo usa salvo que se pase `--seq2seq-ckpt` (así no se muestra texto basura).
- `--clf-ckpt ""` desactiva el clasificador si se quiere probar solo el cableado.

## Bloqueos que quedan

1. **Loader**: `signjoey/dataset.py` + `data.py` usan `torchtext.data.Dataset/Field`
   (API eliminada) y están hardcodeados a CSL-Daily. `pose_dataset.py` es el reemplazo.
   `torchtext` sigue stubbeado porque `vocabulary.py`/`helpers.py` lo importan.
2. **Cabeza LID** (Sign2LID): implementada y cableada desde el loader
   (`test_lid_pipeline.py`). Falta entrenarla con datos de más de una lengua: hoy todos
   los manifests son de 3 columnas y caen en `LID_DEFAULT` (gsg).
3. **Evaluación real**: falta el extractor de features de PHOENIX. El decode actual solo
   valida cableado.

## Lo que estos tests NO prueban

Los landmarks son sintéticos: validan cableado (shapes, máscaras, tres pérdidas, backward,
optimizador, checkpoint), no transferencia. No hay coarticulación, ni gramática facial, ni
variación de firmante, y el texto objetivo es aleatorio.
