/* AgriVision About page: scores, charts, datasets, training info. */
const $ = (s) => document.querySelector(s);

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

function pretty(id) { return id.replace(/___/g, " - ").replace(/_/g, " "); }

async function loadHealth() {
  try { return await api("/api/health"); }
  catch { return null; }
}

async function loadMetrics() {
  const m = await api("/api/metrics");
  $("#modelDesc").textContent = m.model + " · " + (m.note || "");
  const acc = m.test_accuracy * 100;
  $("#accNum").textContent = acc.toFixed(1) + "%";
  $("#valNum").textContent = (m.val_accuracy * 100).toFixed(1) + "%";
  $("#f1Num").textContent = m.macro_avg.f1.toFixed(3);
  $("#wf1Num").textContent = m.weighted_avg.f1.toFixed(3);
  $("#evalSplit").textContent = (m.eval_split || "validation") + " set";

  new Chart($("#accChart"), {
    type: "doughnut",
    data: { labels: ["Correct", "Error"], datasets: [{ data: [m.test_accuracy, 1 - m.test_accuracy], backgroundColor: ["#2E7D32", "#E8DCC8"], borderWidth: 0 }] },
    options: { cutout: "72%", plugins: { legend: { display: false } } },
  });
  $("#metricsTable tbody").innerHTML = m.class_metrics.map((c) => `
    <tr><td><strong>${pretty(c.class)}</strong></td>
    <td class="num">${c.precision.toFixed(3)}</td><td class="num">${c.recall.toFixed(3)}</td>
    <td class="num">${c.f1.toFixed(3)}</td><td class="num">${c.support}</td>
    <td><div class="meter"><div style="width:${(c.f1 * 100).toFixed(1)}%"></div></div></td></tr>`).join("");
  return m;
}

(async () => {
  const h = await loadHealth();
  try {
    await loadMetrics();
    const trained = h && h.loaded;
    $("#statusStrip").innerHTML = trained
      ? '<span class="dot ok"></span><strong>Trained MobileNetV2 weights loaded</strong> - scores below are from real training.'
      : '<span class="dot warn"></span><strong>Fallback demo engine active</strong> - scores below are reference metrics; run backend/train.py on a Kaggle leaf dataset to replace them with real results.';
  } catch (e) { $("#modelDesc").textContent = "API offline: " + e.message; }
})();
