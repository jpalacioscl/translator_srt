#!/bin/bash
# Traductor SRT — lanza llama-server y luego el servidor web FastAPI.
# Uso:  ./run.sh [ruta/al/modelo.gguf]
# Si no se pasa modelo, usa el valor de la variable MODEL o el modelo por defecto.

set -e

DEFAULT_MODEL="/home/jaime/.subtitle_ai/models/Tower-Plus-9B.Q5_K_M.gguf"
MODEL="${1:-${MODEL:-$DEFAULT_MODEL}}"
LLAMACPP_PORT=8080
FASTAPI_PORT=5008
GPU_LAYERS=38        # 42 capas totales; 38 en GPU libera ~600 MB para buffers de inferencia
CONTEXT_SIZE=2048    # 4096 no cabe en 8 GB VRAM con este modelo (9B Q5)

echo "=========================================="
echo "  Traductor SRT con IA Local (llama.cpp)"
echo "=========================================="
echo "  Modelo : $(basename "$MODEL")"
echo "  GPU    : $GPU_LAYERS capas"
echo "  Ctx    : $CONTEXT_SIZE tokens"
echo "=========================================="

# ── 1. Liberar puertos ocupados ───────────────────────────────────────────────
PIDS=$(ss -tlnp sport\ =\ :"$FASTAPI_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
if [[ -n "$PIDS" ]]; then
  echo "⚠  Puerto $FASTAPI_PORT ocupado — matando PID(s): $PIDS"
  kill $PIDS 2>/dev/null || true
  for i in $(seq 1 10); do
    sleep 1
    STILL=$(ss -tlnp sport\ =\ :"$FASTAPI_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
    [[ -z "$STILL" ]] && break
    if [[ $i -eq 10 ]]; then
      echo "   Forzando kill -9 en PID(s): $STILL"
      kill -9 $STILL 2>/dev/null || true
      sleep 1
    fi
  done
  echo "   Puerto $FASTAPI_PORT liberado ✅"
fi

# ── 2. Verificar que el modelo existe ─────────────────────────────────────────
if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: Modelo no encontrado: $MODEL"
  echo ""
  echo "Modelos disponibles:"
  find /home/jaime/.subtitle_ai/models /mnt/datos/llmlocal -name "*.gguf" 2>/dev/null \
    | grep -v "vocab" | sed 's/^/  • /'
  exit 1
fi

# ── 3. Lanzar llama-server si no está corriendo ───────────────────────────────
if curl -s "http://localhost:${LLAMACPP_PORT}/health" > /dev/null 2>&1; then
  echo "✅ llama-server ya está corriendo en puerto $LLAMACPP_PORT"
else
  echo "⚙  Iniciando llama-server..."
  GGML_CUDA_NO_VMM=1 llama-server \
    --model "$MODEL" \
    --n-gpu-layers "$GPU_LAYERS" \
    --ctx-size "$CONTEXT_SIZE" \
    --batch-size 512 \
    --ubatch-size 256 \
    --parallel 1 \
    --host 0.0.0.0 \
    --port "$LLAMACPP_PORT" \
    --log-disable \
    > /tmp/llama-server.log 2>&1 &
  LLAMA_PID=$!
  echo "   PID: $LLAMA_PID (log: /tmp/llama-server.log)"

  # Esperar hasta que el servidor responda (máx. 60 s)
  echo -n "   Esperando llama-server"
  for i in $(seq 1 60); do
    if curl -s "http://localhost:${LLAMACPP_PORT}/health" > /dev/null 2>&1; then
      echo " ✅"
      break
    fi
    echo -n "."
    sleep 1
    if [[ $i -eq 60 ]]; then
      echo " ❌"
      echo "ERROR: llama-server no respondió tras 60 s. Ver /tmp/llama-server.log"
      exit 1
    fi
  done
fi

# ── 4. Lanzar FastAPI ─────────────────────────────────────────────────────────
echo ""
echo "🚀 Servidor web en http://localhost:${FASTAPI_PORT}"
echo "   Presiona Ctrl+C para detener."
echo "=========================================="

cd "$(dirname "$0")"
python3 -m uvicorn main:app --host 0.0.0.0 --port "$FASTAPI_PORT" --reload
