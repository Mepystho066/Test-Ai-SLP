# Datos disponibles para LSC / SLT (verificado 2026-10-30)

Regla de esta tabla: solo lo que se pudo comprobar en la fuente (API de figshare/GitHub,
texto del artículo). Lo no verificado va marcado como tal.

## Corpus de lengua de señas

| Corpus | Qué es | Licencia (verificada) | Sirve para |
|---|---|---|---|
| **LSC50** (U. de los Andes) | 50 señas aisladas, 4 000 vídeos RGB+depth+IR, IMU y landmarks; 5 firmantes | **CC BY 4.0** (figshare DOI 10.6084/m9.figshare.27383016) — uso comercial permitido con atribución | Pre-entrenar el encoder de pose y el modo vocabulario de la app |
| **LSC-54** (U. de La Sabana, Data in Brief 2025) | 54 clases (11 colores + 33 cortesías + números 1-10); 2 850 vídeos (128 250 tras aumento ×45); JSON con `face`, `pose`, `r_hand`, `l_hand` (MediaPipe) + vídeos anonimizados | Pendiente de confirmar en el artículo (acceso abierto) | Ídem; sus grupos coinciden con el contrato de 318 dim |
| PHOENIX-2014T | Frases continuas DGS→alemán | **No comercial** (README de RWTH) | Arquitectura y comparabilidad, no producto |
| CSL-Daily | Frases continuas CSL→chino | No comercial y **prohibida la redistribución** | Validación interna |
| SP-10 (SpreadTheSign) | 10 lenguas, vocabulario | Términos de SpreadTheSign | Validación tras acuerdo |
| YouTube-SL-25 | >3 000 h, >25 lenguas, subtítulos | Solo IDs de vídeo; cada vídeo conserva su licencia | Pre-entrenamiento si se respeta el origen |

## Lo que NO existe (y conviene decir en voz alta)

- **No hay corpus abierto de LSC continua con traducción al español.** LSC50 y LSC-54 son
  señas **aisladas**: entrenan un clasificador/encoder, no un traductor seña→frase.
- **No hay encoder de señas pre-entrenado y abierto para LSC.** SignGemma es ASL→inglés y no
  se publicó. El encoder de pose se entrena desde cero (39-50 M params, una GPU modesta).
- Por eso, en el Hito 1, la traducción se valida en PHOENIX (alemán) y el español se genera
  traduciendo las referencias por MT o con datos propios.

## Cómo encajan en el contrato de pose (`pose_contract.py`)

### LSC50 (verificado descomprimiendo el ZIP, 4 000 clips)

```
LANDMARKS/BODY_LANDMARKS/CCCC_SSSS_RRRR.csv        33 landmarks x3 = 99 cols
LANDMARKS/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS/...  21 landmarks x3 = 63 cols
LANDMARKS/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS/... 21 landmarks x3 = 63 cols
LANDMARKS/FACE_LANDMARKS/...                       468 landmarks x3 = 1404 cols
```

- `CCCC` = clase (0-49), `SSSS` = firmante (0-4), `RRRR` = repetición. **1 000 clips** de
  landmarks = 50 clases × 5 firmantes × 4 repeticiones (los "4 000 vídeos" del paper son
  las grabaciones entre RGB, profundidad e IR, no 4 000 clips de landmarks).
- Cuerpo y manos vienen con el mismo número de frames (59 en la muestra); **la cara viene a
  otro fps (163 frames en el mismo clip)**. Hay que reescalarla por tiempo normalizado:
  interpolando, la nariz (punto 168 del mallado) correlaciona 0,997 con el landmark 0 del
  cuerpo. Recortar filas no sirve.
- Con esto el contrato de 318 dim se llena completo (manos 42 + brazos 14 + labios 40 +
  cejas 10): **sí hay gramática facial**, a diferencia de lo que se supuso al principio.
- Firmantes: 5. Para evaluar de verdad, el split debe ser **por firmante**, no por repetición.

LSC-54 guarda los mismos cuatro grupos de MediaPipe en JSON normalizado (z relativo a muñeca
y cadera), así que también entra en el contrato sin re-extraer vídeo.

## Formato de datos que consume el repo

`data/<split>.tsv` (sin cabecera) + `data/features/<name>.npy` con shape `(T_enc, 318)`:

```
<name>\t<seq_len>\t<texto objetivo en español>
```
