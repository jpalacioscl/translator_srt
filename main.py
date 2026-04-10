"""
Traductor de subtítulos SRT con IA local (Ollama + llama.cpp)
Backend FastAPI con streaming SSE para progreso en tiempo real.
"""
import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import httpx
import srt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
BATCH_SIZE = 8           # Máximo de líneas por lote
BATCH_MAX_CHARS = 600    # Máximo de caracteres totales por lote
MAX_RETRIES = 3
JOBS_DIR = Path("/tmp/srt_translator_jobs")
JOBS_DIR.mkdir(exist_ok=True)

# Mapa de nombre de idioma → código ISO 639-1
LANG_CODES: dict[str, str] = {
    "español":   "es",
    "inglés":    "en",
    "francés":   "fr",
    "alemán":    "de",
    "italiano":  "it",
    "portugués": "pt",
    "chino":     "zh",
    "japonés":   "ja",
    "árabe":     "ar",
    "coreano":   "ko",
    "ruso":      "ru",
}

# Todos los códigos conocidos para detectar sufijos existentes en el nombre
_ALL_LANG_CODES = set(LANG_CODES.values())


def build_output_filename(original_filename: str, target_lang: str) -> str:
    """
    Construye el nombre del archivo traducido.
    - Si el stem termina en _XX o .XX (código de idioma conocido), lo reemplaza.
      Ej: dino_en.srt  → dino_es.srt
      Ej: movie.en.srt → movie.es.srt
    - Si no tiene sufijo de idioma, lo agrega con punto.
      Ej: pelicula.srt → pelicula.es.srt
    """
    lang_code = LANG_CODES.get(target_lang.lower(), target_lang.lower()[:2])
    stem = Path(original_filename).stem   # "dino_en" de "dino_en.srt"

    # Detectar sufijo _XX o .XX al final del stem
    match = re.search(r"([_\.])([a-z]{2,3})$", stem, re.IGNORECASE)
    if match and match.group(2).lower() in _ALL_LANG_CODES:
        # Reemplazar código existente manteniendo el separador original
        separator = match.group(1)
        base = stem[: match.start()]
        new_stem = f"{base}{separator}{lang_code}"
    else:
        # Sin sufijo de idioma: agregar .XX
        new_stem = f"{stem}.{lang_code}"

    return f"{new_stem}.srt"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Traductor SRT Local IA", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Utilidades SRT
# ---------------------------------------------------------------------------

def parse_srt(content: str) -> list[srt.Subtitle]:
    """Parsea el contenido de un archivo SRT y retorna lista de subtítulos."""
    try:
        return list(srt.parse(content))
    except Exception as e:
        raise ValueError(f"Formato SRT inválido: {e}")


def subtitles_to_srt(subtitles: list[srt.Subtitle]) -> str:
    """Convierte lista de subtítulos de vuelta a formato SRT."""
    return srt.compose(subtitles)


def strip_html_tags(text: str) -> str:
    """Extrae texto limpio para traducir (preserva tags al reconstruir)."""
    return re.sub(r"<[^>]+>", "", text).strip()


def extract_html_wrapper(text: str) -> tuple[str, str]:
    """Retorna (prefix_tags, suffix_tags) que envuelven el texto."""
    prefix = re.findall(r"^(<[^>]+>)+", text)
    suffix = re.findall(r"(<[^>]+>)+$", text)
    return ("".join(prefix), "".join(suffix))


# ---------------------------------------------------------------------------
# Cliente Ollama
# ---------------------------------------------------------------------------

async def ollama_translate(
    client: httpx.AsyncClient,
    model: str,
    text_block: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """
    Envía un bloque de texto numerado a Ollama para traducir.
    Retorna el bloque traducido en el mismo formato numerado.
    """
    system_prompt = f"""Eres un traductor profesional especializado en subtítulos de video.
Tu tarea es traducir del {source_lang} al {target_lang}.

REGLAS ESTRICTAS:
1. Traduce ÚNICAMENTE el texto entre los marcadores de número [N].
2. Conserva EXACTAMENTE el mismo formato: [N] texto traducido
3. NO agregues explicaciones, notas ni texto adicional.
4. Preserva nombres propios, marcas y términos técnicos cuando sea apropiado.
5. Mantén el tono, registro y emoción del original.
6. Los subtítulos son líneas breves de diálogo: usa lenguaje natural y coloquial.
7. NO traduzcas los corchetes ni los números.
8. Responde SOLO con las líneas numeradas traducidas, nada más."""

    user_prompt = f"Traduce estas líneas de subtítulo:\n\n{text_block}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,   # Desactiva thinking en qwen3/deepseek-r1 (Ollama >= 0.6)
        "options": {
            "temperature": 0.1,   # Baja temperatura = más consistente
            "top_p": 0.9,
            "num_predict": -1,    # Sin límite de tokens en la respuesta
        },
    }

    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    raw = data["message"]["content"].strip()
    # Eliminar bloques <think>...</think> por si el modelo los incluye igualmente
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


def create_batches(subtitles: list[srt.Subtitle]) -> list[list[srt.Subtitle]]:
    """Crea batches respetando límite de líneas Y de caracteres totales."""
    batches = []
    current: list[srt.Subtitle] = []
    current_chars = 0

    for sub in subtitles:
        sub_chars = len(strip_html_tags(sub.content))
        if current and (len(current) >= BATCH_SIZE or current_chars + sub_chars > BATCH_MAX_CHARS):
            batches.append(current)
            current = [sub]
            current_chars = sub_chars
        else:
            current.append(sub)
            current_chars += sub_chars

    if current:
        batches.append(current)
    return batches


def build_text_block(subtitles: list[srt.Subtitle]) -> str:
    """Crea un bloque numerado de texto para enviar al modelo."""
    lines = []
    for sub in subtitles:
        clean = strip_html_tags(sub.content).replace("\n", " ").strip()
        lines.append(f"[{sub.index}] {clean}")
    return "\n".join(lines)


def parse_translated_block(
    response: str, original_subtitles: list[srt.Subtitle]
) -> dict[int, str]:
    """
    Parsea la respuesta del modelo y retorna {index: translated_text}.
    Es tolerante a variaciones de formato que el modelo pueda producir.
    """
    translations: dict[int, str] = {}
    # Patron flexible: [N] texto  o  N. texto  o  N) texto
    pattern = re.compile(r"^\s*[\[(\{]?(\d+)[\])\}]?[.\-\s]+(.+)$", re.MULTILINE)

    for match in pattern.finditer(response):
        idx = int(match.group(1))
        text = match.group(2).strip()
        # Verificar que el índice pertenece al batch actual
        valid_indices = {s.index for s in original_subtitles}
        if idx in valid_indices:
            translations[idx] = text

    return translations


# ---------------------------------------------------------------------------
# Motor de traducción con SSE
# ---------------------------------------------------------------------------

async def translate_srt_stream(
    job_id: str,
    subtitles: list[srt.Subtitle],
    model: str,
    source_lang: str,
    target_lang: str,
) -> None:
    """
    Traduce todos los subtítulos en batches y guarda el resultado.
    Escribe eventos SSE en un archivo de cola para que el endpoint los sirva.
    """
    queue_file = JOBS_DIR / f"{job_id}.jsonl"
    result_file = JOBS_DIR / f"{job_id}.srt"

    total = len(subtitles)
    translated_subtitles = [
        srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=sub.content)
        for sub in subtitles
    ]
    translated_map = {sub.index: sub for sub in translated_subtitles}

    def emit(event_type: str, data: dict) -> None:
        """Escribe un evento SSE en la cola del job."""
        with open(queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": event_type, "data": data}) + "\n")

    emit("start", {"total": total, "model": model, "target_lang": target_lang})

    async with httpx.AsyncClient() as client:
        # Verificar que Ollama está disponible
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
        except Exception:
            emit("error", {"message": "Ollama no está disponible. Ejecuta: ollama serve"})
            emit("done", {"success": False})
            return

        processed = 0
        batches = create_batches(subtitles)

        for batch_idx, batch in enumerate(batches):
            retries = 0
            success = False
            current_batch = batch

            while retries < MAX_RETRIES and not success:
                try:
                    text_block = build_text_block(current_batch)
                    response_text = await ollama_translate(
                        client, model, text_block, source_lang, target_lang
                    )
                    translations = parse_translated_block(response_text, current_batch)

                    # Verificar cobertura mínima (80% del batch)
                    coverage = len(translations) / len(current_batch)
                    if coverage < 0.8 and retries < MAX_RETRIES - 1:
                        raise ValueError(
                            f"Cobertura insuficiente: {coverage:.0%} ({len(translations)}/{len(current_batch)})"
                        )

                    # Aplicar traducciones
                    for sub in current_batch:
                        if sub.index in translations:
                            prefix, suffix = extract_html_wrapper(sub.content)
                            translated_map[sub.index].content = (
                                prefix + translations[sub.index] + suffix
                            )
                        # Si no se tradujo, se mantiene el original

                    processed += len(current_batch)
                    success = True

                    emit("progress", {
                        "processed": processed,
                        "total": total,
                        "percent": round(processed / total * 100, 1),
                        "batch": batch_idx + 1,
                        "total_batches": len(batches),
                        "sample": translations.get(current_batch[0].index, ""),
                    })

                except Exception as e:
                    retries += 1
                    if retries < MAX_RETRIES:
                        emit("warning", {
                            "message": f"Reintentando batch {batch_idx + 1} (intento {retries}/{MAX_RETRIES}): {str(e)[:100]}"
                        })
                        # En el último reintento, reducir el batch a la mitad
                        if retries == MAX_RETRIES - 1 and len(current_batch) > 1:
                            current_batch = current_batch[: len(current_batch) // 2]
                        await asyncio.sleep(1)
                    else:
                        emit("warning", {
                            "message": f"Batch {batch_idx + 1} falló tras {MAX_RETRIES} intentos. Se mantiene texto original."
                        })
                        processed += len(batch)
                        emit("progress", {
                            "processed": processed,
                            "total": total,
                            "percent": round(processed / total * 100, 1),
                            "batch": batch_idx + 1,
                            "total_batches": len(batches),
                            "sample": "(original conservado)",
                        })

    # Guardar resultado final
    result_srt = subtitles_to_srt(list(translated_map.values()))
    result_file.write_text(result_srt, encoding="utf-8")
    emit("done", {"success": True, "job_id": job_id})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/models")
async def get_models():
    """Obtiene la lista de modelos disponibles en Ollama."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models, "ollama_available": True}
        except Exception:
            return {"models": [], "ollama_available": False}


@app.post("/api/translate")
async def start_translation(
    file: UploadFile = File(...),
    model: str = Form("qwen3:8b"),
    source_lang: str = Form("inglés"),
    target_lang: str = Form("español"),
):
    """Inicia un job de traducción y retorna el job_id."""
    if not file.filename.lower().endswith(".srt"):
        raise HTTPException(400, "Solo se aceptan archivos .srt")

    content_bytes = await file.read()
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = content_bytes.decode("latin-1")

    try:
        subtitles = parse_srt(content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not subtitles:
        raise HTTPException(400, "El archivo SRT no contiene subtítulos válidos")

    job_id = str(uuid.uuid4())[:8]

    output_filename = build_output_filename(file.filename, target_lang)

    # Guardar SRT original y metadatos del job
    (JOBS_DIR / f"{job_id}_original.srt").write_text(content, encoding="utf-8")
    (JOBS_DIR / f"{job_id}_meta.json").write_text(
        json.dumps({"output_filename": output_filename, "target_lang": target_lang}),
        encoding="utf-8",
    )

    # Iniciar traducción en background
    asyncio.create_task(
        translate_srt_stream(job_id, subtitles, model, source_lang, target_lang)
    )

    return {
        "job_id": job_id,
        "total_subtitles": len(subtitles),
        "filename": file.filename,
        "output_filename": output_filename,
    }


@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    """Server-Sent Events: devuelve el progreso de traducción en tiempo real."""
    queue_file = JOBS_DIR / f"{job_id}.jsonl"
    timeout = 600  # 10 minutos máximo

    async def event_generator() -> AsyncGenerator[str, None]:
        start = time.time()
        sent_lines = 0
        done = False

        while not done and (time.time() - start) < timeout:
            if queue_file.exists():
                lines = queue_file.read_text(encoding="utf-8").splitlines()
                for line in lines[sent_lines:]:
                    if line.strip():
                        event = json.loads(line)
                        yield f"data: {json.dumps(event)}\n\n"
                        sent_lines += 1
                        if event["type"] == "done":
                            done = True
                            break
            if not done:
                await asyncio.sleep(0.3)

        if not done:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Timeout: la traducción tardó demasiado'}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Descarga el archivo SRT traducido con el nombre original + código de idioma."""
    result_file = JOBS_DIR / f"{job_id}.srt"
    if not result_file.exists():
        raise HTTPException(404, "Resultado no encontrado o traducción aún en proceso")

    meta_file = JOBS_DIR / f"{job_id}_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        output_filename = meta.get("output_filename", f"subtitulo_{job_id}.srt")
    else:
        output_filename = f"subtitulo_{job_id}.srt"

    return FileResponse(
        str(result_file),
        media_type="text/plain",
        filename=output_filename,
        headers={"Content-Disposition": f'attachment; filename="{output_filename}"'},
    )


@app.get("/api/preview/{job_id}")
async def preview_result(job_id: str):
    """Previsualiza original y traducción lado a lado."""
    original_file = JOBS_DIR / f"{job_id}_original.srt"
    result_file = JOBS_DIR / f"{job_id}.srt"

    if not result_file.exists():
        raise HTTPException(404, "Resultado no disponible aún")

    original_subs = list(srt.parse(original_file.read_text(encoding="utf-8")))
    translated_subs = list(srt.parse(result_file.read_text(encoding="utf-8")))

    preview = []
    for orig, trans in zip(original_subs[:50], translated_subs[:50]):
        preview.append({
            "index": orig.index,
            "time": str(orig.start)[:11],
            "original": orig.content,
            "translated": trans.content,
        })
    return {"preview": preview, "total": len(original_subs)}
