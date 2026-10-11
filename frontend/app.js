/* AgriVision homepage: health badge, hero stats, leaf diagnosis. */
const $ = (s) => document.querySelector(s);
const API = ""; // same origin (Render serves API + frontend together)

let emptyHTML = "";

async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

// ---------- health + hero stats ----------
async function loadHealth() {
  try {
    await api("/api/health");
    const m = await api("/api/metrics");
    if ($("#statAcc")) $("#statAcc").textContent = (m.test_accuracy * 100).toFixed(1) + "%";
    if ($("#statF1")) $("#statF1").textContent = m.macro_avg.f1.toFixed(3);
    if ($("#statClasses")) $("#statClasses").textContent = m.class_metrics.length;
    if ($("#ctaAcc")) $("#ctaAcc").textContent = (m.test_accuracy * 100).toFixed(1) + "% test accuracy";
    if ($("#glAcc")) $("#glAcc").textContent = (m.test_accuracy * 100).toFixed(1) + "%";
    if ($("#glVal")) $("#glVal").textContent = (m.val_accuracy * 100).toFixed(1) + "%";
    if ($("#glF1")) $("#glF1").textContent = m.macro_avg.f1.toFixed(3);
    if ($("#glClasses")) $("#glClasses").textContent = m.class_metrics.length;
    if ($("#ctaSub")) $("#ctaSub").textContent =
      `${m.model.split(",")[0]} · macro F1 ${m.macro_avg.f1.toFixed(3)} · ${m.class_metrics.length} classes - see full scores, per-class metrics, dataset & training details.`;
  } catch { /* API offline: stats keep placeholder values */ }
}

// ---------- upload / predict ----------
function initPredict() {
  const drop = $("#drop"), fileInput = $("#fileInput");
  if (!drop || !fileInput) return;
  emptyHTML = $("#resultEmpty").innerHTML;
  const openPicker = (e) => { if (e) e.stopPropagation(); fileInput.click(); };
  $("#browseBtn").addEventListener("click", openPicker);
  $("#reuploadBtn").addEventListener("click", openPicker);
  $("#clearBtn").addEventListener("click", (e) => { e.stopPropagation(); resetUpload(); });
  drop.addEventListener("click", () => { if (!$("#dropEmpty").hidden) fileInput.click(); });
  ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) runPredict(f);
  });
  fileInput.addEventListener("change", () => { if (fileInput.files[0]) runPredict(fileInput.files[0]); });

  // sample images: real photos from the validation set (frontend/samples/)
  const SAMPLES = { healthy: "healthy.jpg", blight: "blight.jpg", rust: "rust.jpg" };
  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", async () => {
    const kind = b.dataset.sample;
    b.disabled = true;
    try {
      const r = await fetch(`/static/samples/${SAMPLES[kind] || "healthy.jpg"}`);
      if (!r.ok) throw new Error("sample image missing");
      const blob = await r.blob();
      runPredict(new File([blob], kind + ".jpg", { type: "image/jpeg" }));
    } catch (e) {
      $("#resultEmpty").hidden = false;
      $("#resultEmpty").innerHTML = "<p>Could not load sample: " + e.message + "</p>";
    } finally {
      b.disabled = false;
    }
  }));
}

async function runPredict(file) {
  const url = URL.createObjectURL(file);
  $("#preview").src = url;
  $("#dropEmpty").hidden = true;
  $("#dropFilled").hidden = false;
  const hero = $("#heroPreview"); if (hero) hero.src = url;
  $("#resultEmpty").hidden = true; $("#resultBody").hidden = true; $("#loading").hidden = false;
  try {
    const fd = new FormData();
    fd.append("file", file);
    const d = await api("/api/predict", { method: "POST", body: fd });
    showResult(d);
  } catch (e) {
    $("#loading").hidden = true; $("#resultEmpty").hidden = false;
    $("#resultEmpty").innerHTML = "<p>Upload failed: " + e.message + "</p>";
  }
}

function sevChip(sev) {
  const el = $("#severityChip");
  el.textContent = "● " + (sev || "Unknown");
  el.className = "chip" + (sev === "None" ? "" : sev === "Critical" || sev === "High" ? " bad" : " warn");
}

function shortEngine(name) {
  const n = (name || "").toLowerCase();
  if (n.includes("pytorch") && n.includes("cuda")) return "Trained · GPU";
  if (n.includes("pytorch")) return "Trained · CPU";
  if (n.includes("keras")) return "Trained · Keras";
  if (n.includes("fallback")) return "Demo mode";
  return name || "model";
}

function showResult(d) {
  $("#loading").hidden = true; $("#resultBody").hidden = false;
  $("#predName").textContent = d.pretty;
  $("#predCause").textContent = (d.info && d.info.cause) || "";
  $("#treatText").textContent = (d.info && d.info.treatment) || "-";
  $("#disclaimer").textContent = d.disclaimer || "";
  const pct = (d.confidence * 100);
  $("#confNum").textContent = pct.toFixed(1) + "%";
  $("#confBar").style.width = pct + "%";
  const eng = $("#engineChip");
  eng.textContent = shortEngine(d.features && d.features.engine)
    + (d.trained && d.model_accuracy ? ` · ${(d.model_accuracy * 100).toFixed(1)}% val` : "");
  eng.title = (d.features && d.features.engine) || "";
  eng.className = "chip " + (d.trained ? "" : "warn");
  sevChip(d.info && d.info.severity);
  $("#top3").innerHTML = d.top3.map((t) =>
    `<div class="row"><span class="nm" title="${t.label}">${t.label}</span><span class="pc">${(t.confidence * 100).toFixed(1)}%</span><span class="track"><span class="fill" style="display:block;width:${(t.confidence * 100).toFixed(1)}%;height:100%"></span></span></div>`
  ).join("");
  const hp = $("#heroPred");
  if (hp) hp.textContent = d.pretty + " · " + pct.toFixed(1) + "%";
}

function resetUpload() {
  const pv = $("#preview");
  if (pv.src.startsWith("blob:")) URL.revokeObjectURL(pv.src);
  pv.removeAttribute("src");
  $("#dropFilled").hidden = true;
  $("#dropEmpty").hidden = false;
  $("#fileInput").value = "";
  $("#resultBody").hidden = true;
  $("#loading").hidden = true;
  const empty = $("#resultEmpty");
  if (emptyHTML) empty.innerHTML = emptyHTML;
  empty.hidden = false;
  const hero = $("#heroPreview");
  if (hero) hero.src = "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?w=640&q=80";
  const hp = $("#heroPred");
  if (hp) hp.textContent = "Upload a photo to begin";
}

loadHealth(); initPredict();
