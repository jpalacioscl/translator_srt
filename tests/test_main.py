import json
import uuid

import pytest
import srt
from fastapi.testclient import TestClient

import main
from main import (
    build_output_filename,
    build_text_block,
    create_batches,
    extract_html_wrapper,
    parse_srt,
    parse_translated_block,
    strip_html_tags,
    subtitles_to_srt,
)

SAMPLE_SRT = """1
00:00:00,000 --> 00:00:02,000
Hello, how are you?

2
00:00:02,500 --> 00:00:05,000
This is a test subtitle file.

3
00:00:05,500 --> 00:00:08,000
Thank you very much.
"""


# ---------------------------------------------------------------------------
# build_output_filename
# ---------------------------------------------------------------------------

def test_build_output_filename_no_existing_suffix():
    assert build_output_filename("pelicula.srt", "español") == "pelicula.es.srt"


def test_build_output_filename_replaces_underscore_suffix():
    assert build_output_filename("dino_en.srt", "español") == "dino_es.srt"


def test_build_output_filename_replaces_dot_suffix():
    assert build_output_filename("movie.en.srt", "español") == "movie.es.srt"


def test_build_output_filename_unknown_target_lang_uses_first_two_chars():
    assert build_output_filename("clip.srt", "Klingon") == "clip.kl.srt"


# ---------------------------------------------------------------------------
# parse_srt / subtitles_to_srt
# ---------------------------------------------------------------------------

def test_parse_srt_valid_content():
    subs = parse_srt(SAMPLE_SRT)
    assert len(subs) == 3
    assert subs[0].content == "Hello, how are you?"


def test_parse_srt_invalid_content_raises_value_error():
    with pytest.raises(ValueError):
        parse_srt("esto no es un srt valido")


def test_subtitles_to_srt_round_trip():
    subs = parse_srt(SAMPLE_SRT)
    composed = subtitles_to_srt(subs)
    reparsed = parse_srt(composed)
    assert [s.content for s in reparsed] == [s.content for s in subs]


# ---------------------------------------------------------------------------
# strip_html_tags / extract_html_wrapper
# ---------------------------------------------------------------------------

def test_strip_html_tags_removes_tags():
    assert strip_html_tags("<i>Hello</i> world") == "Hello world"


def test_strip_html_tags_plain_text_unchanged():
    assert strip_html_tags("plain text") == "plain text"


def test_extract_html_wrapper_with_tags():
    prefix, suffix = extract_html_wrapper("<i><b>text</b></i>")
    assert prefix == "<i><b>"
    assert suffix == "</b></i>"


def test_extract_html_wrapper_no_tags():
    assert extract_html_wrapper("plain text") == ("", "")


# ---------------------------------------------------------------------------
# create_batches
# ---------------------------------------------------------------------------

def test_create_batches_respects_batch_size():
    subs = parse_srt(SAMPLE_SRT) * 1  # 3 subs, BATCH_SIZE == 3
    batches = create_batches(subs)
    assert sum(len(b) for b in batches) == len(subs)
    assert all(len(b) <= main.BATCH_SIZE for b in batches)


def test_create_batches_splits_on_char_limit():
    long_content = "x" * main.BATCH_MAX_CHARS
    subs = [
        srt.Subtitle(index=1, start=srt.srt_timestamp_to_timedelta("00:00:00,000"),
                     end=srt.srt_timestamp_to_timedelta("00:00:01,000"), content=long_content),
        srt.Subtitle(index=2, start=srt.srt_timestamp_to_timedelta("00:00:01,000"),
                     end=srt.srt_timestamp_to_timedelta("00:00:02,000"), content="short line"),
    ]
    batches = create_batches(subs)
    assert len(batches) == 2


def test_create_batches_empty_input():
    assert create_batches([]) == []


# ---------------------------------------------------------------------------
# build_text_block / parse_translated_block
# ---------------------------------------------------------------------------

def test_build_text_block_sequential_numbering_and_index_map():
    subs = parse_srt(SAMPLE_SRT)
    # simulate non-sequential real indices
    subs[0].index = 10
    subs[1].index = 20
    subs[2].index = 30

    text_block, index_map = build_text_block(subs)

    assert text_block.splitlines() == [
        "[1] Hello, how are you?",
        "[2] This is a test subtitle file.",
        "[3] Thank you very much.",
    ]
    assert index_map == [10, 20, 30]


def test_parse_translated_block_standard_format():
    subs = parse_srt(SAMPLE_SRT)
    _, index_map = build_text_block(subs)
    response = "[1] Hola, ¿cómo estás?\n[2] Este es un archivo de prueba.\n[3] Muchas gracias."

    translations = parse_translated_block(response, subs, index_map)

    assert translations[subs[0].index] == "Hola, ¿cómo estás?"
    assert translations[subs[1].index] == "Este es un archivo de prueba."
    assert translations[subs[2].index] == "Muchas gracias."


@pytest.mark.parametrize("line_format", ["1. texto", "1) texto", "1: texto", "[1] texto"])
def test_parse_translated_block_tolerant_formats(line_format):
    subs = parse_srt(SAMPLE_SRT)[:1]
    _, index_map = build_text_block(subs)
    translations = parse_translated_block(line_format, subs, index_map)
    assert translations[subs[0].index] == "texto"


def test_parse_translated_block_ignores_out_of_range_numbers():
    subs = parse_srt(SAMPLE_SRT)[:1]
    _, index_map = build_text_block(subs)
    translations = parse_translated_block("[99] texto fuera de rango", subs, index_map)
    assert translations == {}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def isolate_llamacpp_base_url(monkeypatch):
    # Ningún test debe depender de si llama-server está corriendo en la máquina.
    monkeypatch.setattr(main, "LLAMACPP_BASE_URL", "http://localhost:1")
    yield


def test_root_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_get_models_when_llamacpp_unavailable(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": [], "llamacpp_available": False}


def test_translate_rejects_non_srt_file(client):
    resp = client.post(
        "/api/translate",
        files={"file": ("archivo.txt", b"hola", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Solo se aceptan archivos .srt" in resp.json()["detail"]


def test_translate_rejects_invalid_srt_content(client):
    resp = client.post(
        "/api/translate",
        files={"file": ("archivo.srt", b"esto no es srt valido", "text/plain")},
    )
    assert resp.status_code == 400
    assert "Formato SRT inv" in resp.json()["detail"]


def test_translate_rejects_empty_srt(client):
    resp = client.post(
        "/api/translate",
        files={"file": ("archivo.srt", b"", "text/plain")},
    )
    assert resp.status_code == 400


def test_translate_success_schedules_job_without_hitting_network(client, monkeypatch):
    started = {}

    async def fake_translate_srt_stream(job_id, subtitles, model, source_lang, target_lang):
        started["job_id"] = job_id

    monkeypatch.setattr(main, "translate_srt_stream", fake_translate_srt_stream)

    resp = client.post(
        "/api/translate",
        files={"file": ("pelicula.srt", SAMPLE_SRT.encode("utf-8"), "text/plain")},
        data={"model": "test-model", "source_lang": "inglés", "target_lang": "español"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_subtitles"] == 3
    assert body["filename"] == "pelicula.srt"
    assert body["output_filename"] == "pelicula.es.srt"
    assert (main.JOBS_DIR / f"{body['job_id']}_original.srt").exists()
    assert (main.JOBS_DIR / f"{body['job_id']}_meta.json").exists()

    (main.JOBS_DIR / f"{body['job_id']}_original.srt").unlink(missing_ok=True)
    (main.JOBS_DIR / f"{body['job_id']}_meta.json").unlink(missing_ok=True)


def test_download_missing_job_returns_404(client):
    resp = client.get(f"/api/download/{uuid.uuid4().hex[:8]}")
    assert resp.status_code == 404


def test_preview_missing_job_returns_404(client):
    resp = client.get(f"/api/preview/{uuid.uuid4().hex[:8]}")
    assert resp.status_code == 404


def test_stream_returns_queued_events_immediately(client):
    job_id = uuid.uuid4().hex[:8]
    queue_file = main.JOBS_DIR / f"{job_id}.jsonl"
    events = [
        {"type": "start", "data": {"total": 1}},
        {"type": "progress", "data": {"processed": 1, "total": 1}},
        {"type": "done", "data": {"success": True, "job_id": job_id}},
    ]
    queue_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    try:
        with client.stream("GET", f"/api/stream/{job_id}") as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert '"type": "start"' in body
        assert '"type": "done"' in body
    finally:
        queue_file.unlink(missing_ok=True)
