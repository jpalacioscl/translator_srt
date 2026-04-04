/**
 * Traductor SRT — Frontend
 * Gestiona: upload, configuración, SSE progress, descarga y vista previa.
 */

const API = "";  // mismo origen

// ─── Estado global ─────────────────────────────────────────────────────────
let currentJobId = null;
let selectedFile = null;
let startTime = null;

// ─── Referencias DOM ────────────────────────────────────────────────────────
const ollamaStatus   = document.getElementById("ollamaStatus");
const modelSelect    = document.getElementById("modelSelect");
const sourceLang     = document.getElementById("sourceLang");
const targetLang     = document.getElementById("targetLang");
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const fileInfo       = document.getElementById("fileInfo");
const fileName       = document.getElementById("fileName");
const fileSize       = document.getElementById("fileSize");
const clearFile      = document.getElementById("clearFile");
const translateBtn   = document.getElementById("translateBtn");
const progressCard   = document.getElementById("progressCard");
const progressLabel  = document.getElementById("progressLabel");
const progressPct    = document.getElementById("progressPercent");
const progressFill   = document.getElementById("progressFill");
const progressBatch  = document.getElementById("progressBatch");
const progressEta    = document.getElementById("progressEta");
const sampleText     = document.getElementById("sampleText");
const warningList    = document.getElementById("warningList");
const resultCard     = document.getElementById("resultCard");
const downloadBtn    = document.getElementById("downloadBtn");
const previewBtn     = document.getElementById("previewBtn");
const newBtn         = document.getElementById("newTranslationBtn");
const previewCard    = document.getElementById("previewCard");
const previewBody    = document.getElementById("previewBody");
const closePreview   = document.getElementById("closePreview");

// ─── Init ────────────────────────────────────────────────────────────────────
(async () => {
  await checkOllama();
})();

// ─── Verificar Ollama ────────────────────────────────────────────────────────
async function checkOllama() {
  try {
    const res = await fetch(`${API}/api/models`);
    const data = await res.json();

    if (data.ollama_available && data.models.length > 0) {
      setStatus("ok", `Ollama listo · ${data.models.length} modelos`);
      populateModels(data.models);
    } else if (data.ollama_available) {
      setStatus("warning", "Ollama disponible pero sin modelos");
    } else {
      setStatus("error", "Ollama no disponible · ejecuta: ollama serve");
    }
  } catch {
    setStatus("error", "No se puede conectar al backend");
  }
}

function setStatus(type, text) {
  ollamaStatus.className = `status-badge status-${type}`;
  ollamaStatus.innerHTML = `<span class="dot"></span> ${text}`;
}

function populateModels(models) {
  modelSelect.innerHTML = "";
  const preferred = ["qwen3:8b", "qwen2.5:7b", "llama3.2:3b"];
  const sorted = [...models].sort((a, b) => {
    const ai = preferred.indexOf(a), bi = preferred.indexOf(b);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return a.localeCompare(b);
  });

  sorted.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === "qwen3:8b") opt.textContent += " ★ recomendado";
    modelSelect.appendChild(opt);
  });
}

// ─── Drop Zone ───────────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

clearFile.addEventListener("click", resetFile);

function setFile(file) {
  if (!file.name.toLowerCase().endsWith(".srt")) {
    alert("Por favor selecciona un archivo .srt");
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  dropZone.classList.add("hidden");
  fileInfo.classList.remove("hidden");
  translateBtn.disabled = false;
}

function resetFile() {
  selectedFile = null;
  fileInput.value = "";
  dropZone.classList.remove("hidden");
  fileInfo.classList.add("hidden");
  translateBtn.disabled = true;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// ─── Traducción ──────────────────────────────────────────────────────────────
translateBtn.addEventListener("click", startTranslation);

async function startTranslation() {
  if (!selectedFile) return;

  const model  = modelSelect.value;
  const source = sourceLang.value;
  const target = targetLang.value;

  if (source === target) {
    alert("El idioma origen y destino no pueden ser el mismo.");
    return;
  }

  translateBtn.disabled = true;
  translateBtn.innerHTML = `<span class="btn-icon">⏳</span> Iniciando…`;

  resultCard.classList.add("hidden");
  previewCard.classList.add("hidden");
  progressCard.classList.remove("hidden");
  warningList.innerHTML = "";
  sampleText.textContent = "";
  setProgress(0, "Conectando con Ollama…", "", "");
  startTime = Date.now();

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("model", model);
  formData.append("source_lang", source);
  formData.append("target_lang", target);

  try {
    const res = await fetch(`${API}/api/translate`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Error al iniciar la traducción");
    }

    const data = await res.json();
    currentJobId = data.job_id;
    listenProgress(data.job_id, data.total_subtitles);
  } catch (err) {
    alert(`Error: ${err.message}`);
    resetTranslateBtn();
  }
}

function listenProgress(jobId, total) {
  const es = new EventSource(`${API}/api/stream/${jobId}`);

  es.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "start") {
      setProgress(0, `Traduciendo ${total} subtítulos con ${msg.data.model}…`, "", "");
    }

    if (msg.type === "progress") {
      const d = msg.data;
      const elapsed = (Date.now() - startTime) / 1000;
      const rate = d.processed / elapsed;
      const remaining = rate > 0 ? (total - d.processed) / rate : 0;
      const etaStr = remaining > 5 ? `~${Math.round(remaining)}s restantes` : "casi listo…";

      setProgress(
        d.percent,
        `${d.processed} / ${total} subtítulos`,
        `Lote ${d.batch} / ${d.total_batches}`,
        etaStr,
      );
      if (d.sample) sampleText.textContent = d.sample;
    }

    if (msg.type === "warning") {
      addWarning(msg.data.message);
    }

    if (msg.type === "error") {
      es.close();
      alert(`Error durante la traducción: ${msg.data.message}`);
      resetTranslateBtn();
    }

    if (msg.type === "done" && msg.data.success) {
      es.close();
      setProgress(100, "¡Traducción completada!", "", "");
      progressCard.querySelector("h2.section-title").textContent = "Traducción completa ✅";
      resultCard.classList.remove("hidden");
      translateBtn.innerHTML = `<span class="btn-icon">⚡</span> Traducir con IA`;
      translateBtn.disabled = false;
    }
  };

  es.onerror = () => {
    es.close();
    addWarning("Conexión SSE interrumpida. Verifica si el servidor sigue activo.");
    resetTranslateBtn();
  };
}

function setProgress(pct, label, batch, eta) {
  progressFill.style.width = `${pct}%`;
  progressPct.textContent = `${pct}%`;
  progressLabel.textContent = label;
  progressBatch.textContent = batch;
  progressEta.textContent = eta;
}

function addWarning(msg) {
  const div = document.createElement("div");
  div.className = "warning-item";
  div.textContent = `⚠ ${msg}`;
  warningList.appendChild(div);
}

function resetTranslateBtn() {
  translateBtn.innerHTML = `<span class="btn-icon">⚡</span> Traducir con IA`;
  translateBtn.disabled = false;
}

// ─── Descarga ────────────────────────────────────────────────────────────────
downloadBtn.addEventListener("click", () => {
  if (!currentJobId) return;
  const a = document.createElement("a");
  a.href = `${API}/api/download/${currentJobId}`;
  a.download = `traducido_${currentJobId}.srt`;
  a.click();
});

// ─── Vista previa ────────────────────────────────────────────────────────────
previewBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  try {
    const res = await fetch(`${API}/api/preview/${currentJobId}`);
    const data = await res.json();
    renderPreview(data.preview);
    previewCard.classList.remove("hidden");
    previewCard.scrollIntoView({ behavior: "smooth" });
  } catch {
    alert("Error al cargar la vista previa.");
  }
});

closePreview.addEventListener("click", () => previewCard.classList.add("hidden"));

function renderPreview(rows) {
  previewBody.innerHTML = "";
  rows.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="col-idx">${row.index}</td>
      <td class="col-time">${row.time}</td>
      <td class="col-orig">${escapeHtml(row.original)}</td>
      <td class="col-trans">${escapeHtml(row.translated)}</td>
    `;
    previewBody.appendChild(tr);
  });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

// ─── Nueva traducción ────────────────────────────────────────────────────────
newBtn.addEventListener("click", () => {
  currentJobId = null;
  resetFile();
  resultCard.classList.add("hidden");
  progressCard.classList.add("hidden");
  previewCard.classList.add("hidden");
  progressCard.querySelector("h2.section-title").textContent = "Traduciendo…";
  warningList.innerHTML = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
});
