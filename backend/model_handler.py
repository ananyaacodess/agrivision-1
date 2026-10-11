"""Inference handler: uses trained PyTorch weights (.pt) if present,
legacy Keras weights (.h5) if present, else a transparent heuristic fallback
so the demo + Render deploy work without any weights.

backend/train.py (PyTorch) saves model/plant_disease_mobilenet.pt and
model/metrics.json - picked up automatically on restart.
"""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "model"
LABELS_PATH = MODEL_DIR / "labels.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
PT_WEIGHTS = MODEL_DIR / "plant_disease_mobilenet.pt"
H5_CANDIDATES = [
    MODEL_DIR / "plant_disease_mobilenet.h5",
    MODEL_DIR / "plant_disease_mobilenet.keras",
]

# ImageNet stats (torchvision convention) - must match backend/train.py
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

with open(LABELS_PATH) as f:
    LABELS = json.load(f)

_torch_model = None  # (model, img_size, device)
_tf_model = None
_tf_error = None


def build_mobilenet(num_classes):
    """MobileNetV2 + Dropout(0.3) + Dense-256 + Dropout(0.2) + softmax head.
    Architecture must stay identical to backend/train.py."""
    from torchvision import models
    m = models.mobilenet_v2(weights=None)
    import torch.nn as nn
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return m


def _try_load_torch_model():
    global _torch_model
    if _torch_model is not None:
        return _torch_model
    if not PT_WEIGHTS.exists():
        return None
    try:
        import torch
        ckpt = torch.load(str(PT_WEIGHTS), map_location="cpu", weights_only=True)
        classes = ckpt.get("classes", LABELS)
        img = int(ckpt.get("img", 224))
        model = build_mobilenet(len(classes))
        model.load_state_dict(ckpt["state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        _torch_model = (model, img, device, classes)
        return _torch_model
    except Exception:  # noqa: BLE001 - corrupt weights: fall through to next engine
        return None


def _try_load_tf_model():
    global _tf_model, _tf_error
    if _tf_model is not None:
        return _tf_model
    for p in H5_CANDIDATES:
        if p.exists():
            try:
                import tensorflow as tf  # lazy import, optional at runtime
                _tf_model = tf.keras.models.load_model(str(p))
                return _tf_model
            except Exception as e:  # noqa: BLE001
                _tf_error = str(e)
                return None
    return None


def get_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def model_status():
    if PT_WEIGHTS.exists() and _try_load_torch_model() is not None:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return {"engine": f"mobilenetv2-trained (pytorch, {dev})", "loaded": True}
    if any(p.exists() for p in H5_CANDIDATES) and _try_load_tf_model() is not None:
        return {"engine": "mobilenetv2-trained (keras)", "loaded": True}
    reason = _tf_error or "no trained weights yet - run backend/train.py"
    if PT_WEIGHTS.exists():
        reason = "weights present but could not be loaded (missing torch?)"
    return {"engine": "fallback-heuristic", "loaded": False, "reason": reason}


def _heuristic_predict(arr: np.ndarray):
    """Color-symptom heuristic. Deterministic, explainable, demo-grade."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_mask = (g > r + 0.04) & (g > b + 0.02) & (g > 0.25)
    yellow_mask = (r > 0.45) & (g > 0.4) & (b < 0.35) & (~green_mask)
    brown_mask = (r > 0.25) & (r < 0.65) & (g < 0.5) & (b < 0.35) & (r > b)
    dark_mask = (r < 0.25) & (g < 0.3) & (b < 0.25)
    green_ratio = float(green_mask.mean())
    lesion_ratio = float((yellow_mask | brown_mask).mean())
    dark_ratio = float(dark_mask.mean())
    brightness = float(arr.mean())

    scores = np.full(len(LABELS), 0.05)
    idx = {c: i for i, c in enumerate(LABELS)}

    def add(cls, s):
        if cls in idx:
            scores[idx[cls]] += s

    if lesion_ratio < 0.03 and green_ratio > 0.35:
        for h in [c for c in LABELS if "healthy" in c]:
            add(h, 0.6)
        add("Tomato___healthy", green_ratio)
        add("Apple___healthy", 0.2)
    elif dark_ratio > 0.12 or lesion_ratio > 0.25:
        add("Potato___Late_blight", dark_ratio * 6 + lesion_ratio * 2)
        add("Tomato___Late_blight", dark_ratio * 5 + lesion_ratio * 2)
        add("Tomato___Early_blight", lesion_ratio * 4)
        add("Potato___Early_blight", lesion_ratio * 3.5)
    elif lesion_ratio > 0.08:
        add("Apple___Apple_scab", lesion_ratio * 5)
        add("Apple___Black_rot", lesion_ratio * 4)
        add("Grape___Black_rot", lesion_ratio * 4)
        add("Pepper,_bell___Bacterial_spot", lesion_ratio * 3)
        add("Corn_(maize)___Common_rust_", lesion_ratio * 3)
    else:
        add("Tomato___Early_blight", 0.5 + lesion_ratio * 3)
        add("Potato___Early_blight", 0.4 + lesion_ratio * 3)
        add("Apple___Apple_scab", 0.3)

    if brightness < 0.3:
        add("Potato___Late_blight", 0.5)
        add("Tomato___Late_blight", 0.4)

    scaled = (scores - scores.max()) * 3.0
    e = np.exp(scaled)
    probs = e / e.sum()
    top = int(probs.argmax())
    return top, probs.tolist(), {
        "green_ratio": round(green_ratio, 3),
        "lesion_ratio": round(lesion_ratio, 3),
        "dark_ratio": round(dark_ratio, 3),
        "brightness": round(brightness, 3),
    }


def _torch_predict(image_bytes: bytes):
    loaded = _try_load_torch_model()
    if loaded is None:
        return None
    import torch
    from torchvision import transforms
    model, img_size, device, classes = loaded
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].cpu().numpy()
    if len(classes) != len(LABELS):
        return None  # label drift: let caller fall back
    top = int(probs.argmax())
    return top, [float(p) for p in probs], {
        "engine": f"mobilenetv2-trained (pytorch, {device.type})"}


def _tf_predict(arr: np.ndarray):
    model = _try_load_tf_model()
    if model is None:
        return None
    import tensorflow as tf  # noqa: F811
    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        (arr * 255.0).astype(np.float32))
    x = np.expand_dims(x, 0)
    probs = model.predict(x, verbose=0)[0]
    if len(probs) != len(LABELS):
        return None
    top = int(np.argmax(probs))
    return top, [float(p) for p in probs], {"engine": "mobilenetv2-trained (keras)"}


def predict_image(image_bytes: bytes):
    res = _torch_predict(image_bytes)
    if res is not None:
        return res
    from PIL import Image as _I  # local alias, already imported
    img = _I.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    res = _tf_predict(arr)
    if res is not None:
        return res
    top, probs, feats = _heuristic_predict(arr)
    feats["engine"] = "fallback-heuristic"
    return top, probs, feats


def top_k(probs, k=3):
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return [(LABELS[i], probs[i]) for i in order[:k]]
