#!/bin/bash
# Script de inicio del Traductor SRT
# Verifica que Ollama esté corriendo y lanza el servidor web.

set -e

echo "=========================================="
echo "  Traductor SRT con IA Local"
echo "=========================================="

# 1. Verificar Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "⚠  Ollama no está corriendo. Iniciando..."
  ollama serve &
  sleep 3
  echo "✅ Ollama iniciado."
else
  echo "✅ Ollama ya está corriendo."
fi

# 2. Mostrar modelos disponibles
echo ""
echo "Modelos disponibles:"
curl -s http://localhost:11434/api/tags | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"  • {m['name']}\")
" 2>/dev/null || echo "  (no se pudo listar)"

echo ""
echo "🚀 Iniciando servidor web en http://localhost:8000"
echo "   Abre tu navegador en: http://localhost:8000"
echo ""
echo "   Presiona Ctrl+C para detener."
echo "=========================================="

# 3. Lanzar FastAPI
cd "$(dirname "$0")"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
