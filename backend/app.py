"""AgriVision API - FastAPI backend serving frontend + inference + metrics."""
from pathlib import Path

import json

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model_handler import (
    LABELS, get_metrics, model_status, predict_image, top_k,
    _try_load_torch_model,
)
from .disease_info import get_info
from .datasets import DATASETS, KAGGLE_SETUP

BASE = Path(__file__).resolve().parent.parent
FRONTEND = BASE / "frontend"
MODEL_DIR = BASE / "model"

app = FastAPI(title="AgriVision - Crop Disease Detection", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Fail-fast visibility: preload best weights at startup so logs prove
# which engine serves traffic (trained PyTorch > legacy Keras > heuristic).
_try_load_torch_model()
print("AgriVision engine:", model_status(), flush=True)


@app.get("/api/health")
def health():
    return {"status": "ok", **model_status(), "num_classes": len(LABELS)}


@app.get("/api/classes")
def classes():
    out = []
    for c in LABELS:
        info = get_info(c)
        pretty = c.replace("___", " - ").replace("_", " ")
        out.append({"id": c, "label": pretty, **info})
    return {"classes": out}


@app.get("/api/metrics")
def metrics():
    m = get_metrics()
    m["status"] = model_status()
    return m


@app.get("/api/datasets")
def datasets():
    return {"datasets": DATASETS, "kaggle_setup": KAGGLE_SETUP,
            "status": model_status()}


@app.get("/api/training-status")
def training_status():
    """Live training progress written by backend/train.py (polled by the UI)."""
    p = MODEL_DIR / "training_progress.json"
    if not p.exists():
        return {"running": False, "message": "No training run recorded yet."}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"running": False, "message": "Progress file unreadable."}


@app.get("/api/plots/{name}")
def training_plot(name: str):
    """Training artifacts: training_curves | confusion_matrix (PNG)."""
    allowed = {"training_curves": "training_curves.png",
               "confusion_matrix": "confusion_matrix.png"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Unknown plot.")
    p = MODEL_DIR / allowed[name]
    if not p.exists():
        raise HTTPException(status_code=404, detail="Plot not generated yet - train the model first.")
    return FileResponse(str(p), media_type="image/png")


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 8 MB).")
    try:
        top, probs, feats = predict_image(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    label = LABELS[top]
    info = get_info(label)
    ranked = [{"class": c, "label": c.replace("___", " - ").replace("_", " "),
               "confidence": round(float(p), 4)} for c, p in top_k(probs, 3)]
    engine = (feats.get("engine", "") if isinstance(feats, dict) else "")
    trained = engine.startswith("mobilenetv2-trained")
    try:
        model_acc = float(get_metrics().get("test_accuracy", 0.0))
    except Exception:
        model_acc = 0.0
    if trained:
        disclaimer = (f"Analyzed with the trained model "
                      f"(validation accuracy {model_acc * 100:.1f}%). "
                      f"Confirm with an agronomist before spraying.")
    else:
        disclaimer = ("Demo-grade heuristic prediction (no trained weights loaded). "
                      "Confirm with an agronomist before spraying.")
    return {
        "prediction": label,
        "pretty": label.replace("___", " - ").replace("_", " "),
        "confidence": round(float(probs[top]), 4),
        "top3": ranked,
        "info": info,
        "features": feats,
        "trained": trained,
        "model_accuracy": round(model_acc, 4),
        "disclaimer": disclaimer,
    }


# ---- static frontend (must be last) ----
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/about")
    def about():
        return FileResponse(str(FRONTEND / "about.html"))
