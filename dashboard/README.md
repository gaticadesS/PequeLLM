# Dashboard de chat — PequeLLM

App **Streamlit** que envuelve un checkpoint entrenado en una UI tipo ChatGPT
para poder demostrarlo desde cualquier lado.

> ⚠️ El modelo **no conversa**: hace *completación de tokens*. Escribes un inicio
> de texto y el modelo lo continúa. No tiene memoria (la conversación de varios
> turnos es opcional y sólo concatena texto como contexto).

> 🔒 **Solo-lectura sobre el modelo.** El dashboard nunca escribe el `.pth`: solo
> hace `load_model()` → `eval()` → genera. Los únicos comandos de `run.sh` que
> sobreescriben el checkpoint son `train` y el default sin argumentos.

## Modelos afinados (instrucciones) — flujo de demo

El selector escanea `*.pth` de forma **recursiva** en `/workspace/data`, así que los
modelos afinados que el fine-tuning guarda en `artifacts_instruction/<run>/best_instruction_checkpoint.pth`
**aparecen solos** en el dropdown (etiquetados como *"Instruct ES (afinado) — <run>"*).

Al elegir uno, el toggle **"Modo instrucción"** se activa automáticamente: el chat
arma el prompt con el formato `### Instrucción / ### Respuesta` (el mismo del
fine-tuning) y trata cada mensaje como una instrucción independiente.

Flujo el día del demo:
1. `Ctrl-c` al fine-tuning para detenerlo (el mejor checkpoint ya está guardado).
2. Levantar/refrescar el dashboard (`./run.sh dashboard`) — como el código se monta
   en vivo, basta refrescar el navegador para que aparezca el modelo afinado.
3. Elegir el checkpoint *"Instruct ES (afinado)"* en el sidebar y chatear en español.

## Cómo correr en renna (vía SSH)

```bash
# 1. (solo la primera vez) reconstruir la imagen para que tenga Streamlit
./run.sh build

# 2. (recomendado) respaldar el checkpoint, no se puede reentrenar
cp /workspace/data/pequellm_*.pth ~/backup_checkpoints/   # o cópialo fuera por SSH

# 3. levantar el dashboard (puerto 8501)
./run.sh dashboard
```

El selector del sidebar lista automáticamente los `.pth` que encuentre en
`/workspace/data` (configurable con la env `PEQUELLM_DATA_DIR`) y en la raíz del
repo, etiquetándolos como Small / Medium.

## Verlo desde tu laptop

### Opción A — Probar rápido por SSH (sin túnel)
Reenvía el puerto por SSH y abre el navegador local:

```bash
ssh -L 8501:localhost:8501 usuario@renna
# luego abre http://localhost:8501 en tu laptop
```

### Opción B — Exponerlo a internet con cloudflared (recomendado)
`cloudflared` es **un solo binario, sin root, sin cuenta**. En renna:

```bash
# descargar el binario al home (una vez)
curl -L -o ~/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/cloudflared

# con el dashboard corriendo en otra terminal:
~/cloudflared tunnel --url http://localhost:8501
```

Imprime una URL pública aleatoria (`https://algo.trycloudflare.com`) accesible
desde cualquier dispositivo mientras el túnel esté abierto.

### Opción C — Alternativa con ngrok
Requiere una cuenta gratis (para el authtoken), también es un binario sin root:

```bash
~/ngrok config add-authtoken <TU_TOKEN>
~/ngrok http 8501
```

## Notas
- La GPU AMD solo funciona dentro del contenedor ROCm, por eso Streamlit corre
  dentro del contenedor (vía `run.sh dashboard`), no en el host.
- El puerto se puede cambiar con `DASHBOARD_PORT=9000 ./run.sh dashboard`.
- Si cambias `dashboard/app.py`, **no** necesitas reconstruir la imagen (el repo
  se monta en runtime); solo reinicia `./run.sh dashboard`.
