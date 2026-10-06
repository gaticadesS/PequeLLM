# Embeddings - guia de archivos

Esta carpeta contiene una progresion didactica para construir un mini GPT desde componentes basicos.
Todos los scripts usan PyTorch y leen datos desde `train.bin` con `np.memmap`.

## Requisitos

- Tener `train.bin` en la raiz del repo (`C:\Repos\PequeLLM\train.bin`).
- Tener instaladas dependencias de `requirements.txt` (al menos `torch`, `numpy`, `tokenizers`).
- Ejecutar desde la raiz del proyecto para que la ruta `DATA_PATH = "train.bin"` funcione.
- Para scripts de comparacion: instalar `sentence-transformers`, `scikit-learn` y `matplotlib` si no estan ya en el entorno.
- Para mixed precision en GPU: usar `--precision auto|bf16|fp16` en `emb_gpt2.py`.

## Orden recomendado de estudio

1. `emb_ingenuo.py`
2. `emb_attention.py`
3. `emb_multi-attention.py`
4. `emb_bucle.py`
5. `emb_bb.py`
6. `emb_gpt2.py`
7. `GPT2_DIAGNOSTICS.md`
8. `presentation_report.py`
9. `generate_prompt.py`

## Explicacion de cada archivo

### `emb_ingenuo.py`

- Que hace:
  - Implementa un modelo de lenguaje muy basico con:
    - embedding de token
    - embedding posicional
    - capa lineal de salida (`lm_head`)
  - No usa atencion todavia.
  - Hace un solo `forward pass` y reporta `loss`.
- Para que sirve:
  - Entender el pipeline minimo `x -> embeddings -> logits -> cross_entropy`.
- Nota importante:
  - Tiene `VOCAB_SIZE = 50000`, pero tus bins actuales llegan hasta `50256`, asi que puede dar `IndexError` en algunos lotes.

### `emb_attention.py`

- Que hace:
  - Agrega una cabeza de self-attention causal (`Head`) sobre los embeddings.
  - Mantiene un modelo simple (`GPTBaseModel`) con una sola cabeza.
  - Ejecuta prueba unica de `forward` y `loss`.
- Para que sirve:
  - Entender como un token "atiende" a tokens previos con mascara triangular causal.
- Nota importante:
  - Igual que el anterior, usa `VOCAB_SIZE = 50000`.

### `emb_multi-attention.py`

- Que hace:
  - Escala a una arquitectura tipo Transformer:
    - `MultiHeadAttention`
    - `FeedForward`
    - `Block` con residuales + `LayerNorm`
  - Arma un `GPTModel` con 1 bloque y prueba un `forward`.
- Para que sirve:
  - Ver el salto desde "una cabeza" a un bloque Transformer mas cercano a GPT.
- Nota importante:
  - Sigue con `VOCAB_SIZE = 50000`.

### `emb_bucle.py`

- Que hace:
  - Usa arquitectura tipo GPT (similar a `emb_multi-attention.py`) pero ya con bucle de entrenamiento real.
  - Detecta dispositivo (`cuda`/`cpu`), crea `optimizer`, corre `max_iters`, e imprime loss periodicamente.
- Para que sirve:
  - Primer script de la carpeta que entrena de punta a punta.
- Punto fuerte:
  - Corrige el vocabulario a `VOCAB_SIZE = 65536` para compatibilidad con datos `uint16`.

### `emb_bb.py`

- Que hace:
  - Parte del modelo de `emb_bucle.py` y agrega `generate()` para inferencia autoregresiva.
  - Al final del entrenamiento intenta decodificar texto con `tokenizer-culturax-es-hf.json`.
- Para que sirve:
  - Cerrar el ciclo completo: entrenamiento + generacion de texto.
- Punto fuerte:
  - Tambien usa `VOCAB_SIZE = 65536`.

### `emb_gpt2.py`

- Que hace:
  - Implementa el modelo GPT-2 Small (ver [Emb_gptMed.py](Emb_gptMed.py) para la
    version Medium), con arquitectura instrumentada configurable por CLI:
    - `n_embd = 768`
    - `block_size = 128`
    - `n_layer = 12`
    - `n_head = 12`
  - Incluye entrenamiento instrumentado:
    - train/val loss
    - norma global de gradiente
    - norma de gradiente por capa
    - snapshots UMAP/PCA por iteraciones
    - heuristica de grokking
  - Reanuda automaticamente si existe checkpoint (se puede desactivar con `--no-resume`).
  - Al final genera texto con `generate()`.
- Para que sirve:
  - Pasar de un prototipo didactico a un entrenamiento mas serio, reproducible y medible.
- Archivos relacionados:
  - Checkpoint esperado: `pequellm_gpt2small_checkpoint.pth`
  - Tokenizador esperado: `tokenizer-culturax-es-hf.json`
  - Carpeta de resultados: `Embeddings/artifacts_gpt2/<run_name>`

### `GPT2_DIAGNOSTICS.md`

- Que hace:
  - Explica como mapear cada requerimiento de analisis (gradientes, UMAP, grokking, etc.) a archivos concretos producidos por `emb_gpt2.py`.
- Para que sirve:
  - Tener una guia clara para revisar avances de forma tecnica con su jefe.

### `presentation_report.py`

- Que hace:
  - Construye automaticamente una presentacion en PDF (estilo diapositivas) con:
    - resumen ejecutivo
    - curvas train/val y learning rate
    - analisis de gradientes (global + por capa)
    - snapshots UMAP/PCA por iteracion
    - validacion de embeddings y heuristica de grokking
- Para que sirve:
  - Tener material listo para exponer sin armar diapositivas manualmente despues de entrenar.

### `generate_prompt.py`

- Que hace:
  - Carga un checkpoint de `emb_gpt2.py`.
  - Carga el tokenizador usado por el modelo.
  - Recibe una frase inicial (`--prompt`).
  - Genera varias continuaciones autoregresivas del texto.
- Para que sirve:
  - Probar rapidamente si el modelo ya puede continuar frases de forma cualitativa.
  - Comparar como cambia la generacion al variar `temperature`, `top_k` o el checkpoint.
- Ejemplo:

```powershell
python Embeddings/generate_prompt.py `
  --prompt "Hola, soy un modelo de inteligencia artificial" `
  --num-samples 3 `
  --max-new-tokens 60
```

- Ejemplo guardando resultados:

```powershell
python Embeddings/generate_prompt.py `
  --prompt "Hola, soy un modelo de inteligencia artificial" `
  --num-samples 3 `
  --output-file Embeddings/generations_hola.json
```

- Parametros utiles:
  - `--checkpoint-path`: checkpoint a cargar.
  - `--tokenizer-path`: tokenizador compatible con el checkpoint.
  - `--temperature`: creatividad/ruido de muestreo; `0` usa argmax.
  - `--top-k`: limita el muestreo a los K tokens mas probables.
  - `--max-new-tokens`: cuantos tokens nuevos generar.

## Ejecucion rapida

Desde `C:\Repos\PequeLLM`:

```powershell
python Embeddings/emb_ingenuo.py
python Embeddings/emb_attention.py
python Embeddings/emb_multi-attention.py
python Embeddings/emb_bucle.py
python Embeddings/emb_bb.py
python Embeddings/emb_gpt2.py
python Embeddings/generate_prompt.py --prompt "Hola, soy un modelo de inteligencia artificial"
```

Al terminar `emb_gpt2.py`, se genera automaticamente:

- `Embeddings/artifacts_gpt2/<run_name>/<run_name>_presentation.pdf`
- `Embeddings/artifacts_gpt2/<run_name>/presentation_summary.md`

Tambien puedes regenerar la presentacion de una corrida existente:

```powershell
python Embeddings/presentation_report.py --run-dir Embeddings/artifacts_gpt2/<run_name>
```

Nota sobre tipos numericos:

- IDs de tokens: mantener enteros (`uint16` en binario, `int64` al alimentar `nn.Embedding`).
- Pesos y activaciones del modelo: ahi si conviene usar mixed precision (`fp16`/`bf16`) para rendimiento.

## Observaciones tecnicas

- Todos los scripts comparten la misma funcion `get_batch` basada en `np.memmap`, lo cual es eficiente en RAM.
- Para consistencia con tus datos actuales, conviene unificar todos a `VOCAB_SIZE = 65536` o derivarlo automaticamente del tokenizador/dataset.
- Estos scripts son excelentes para aprendizaje incremental: cada archivo agrega exactamente una idea nueva.
