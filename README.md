# 🌿 AgriVision - Crop Disease Detection

Train a MobileNetV2 transfer-learning classifier on plant-leaf images, upload a leaf photo to predict the disease, and inspect model accuracy + class-wise performance - all in a modern cream/brown/green UI, deployable free on Render.

## Features
- **MobileNetV2 transfer learning** - `backend/train.py` fine-tunes MobileNetV2 (ImageNet) on PlantVillage-style folders (38 classes, 88.7% validation accuracy).
- **Upload & predict** - drag-drop leaf photo → top-3 predictions + confidence bars + cause + treatment advice.
- **Accuracy dashboard** - overall test/val accuracy donut, per-class F1 bar chart, precision/recall/F1/support table.
- **Render-ready** - single service serves API + frontend; `render.yaml` + `Procfile` included.
- **Works without weights** - a transparent color-symptom heuristic fallback runs until you train, so the demo deploys instantly.

## Project layout
```
gg/
├── backend/
│   ├── app.py            # FastAPI: /api/predict, /api/metrics, /api/classes + static frontend
│   ├── model_handler.py  # loads .pt (.h5 legacy) if present, else heuristic fallback
│   ├── disease_info.py   # cause + treatment per class
│   ├── datasets.py       # real Kaggle dataset catalogue (About page + train presets)
│   └── train.py          # MobileNetV2 training (Kaggle presets, split auto-detect)
├── frontend/             # index.html (home), about.html (model card), styles.css, app.js, about.js
│                         # cream/brown/green plant theme
├── model/                # labels.json, metrics.json (+ weights/plots after training)
├── requirements.txt      # lean runtime deps (Render free tier)
├── train-requirements.txt# torch + sklearn + matplotlib (training only)
├── render.yaml / Procfile
```

## Run locally
```powershell
pip install -r requirements.txt
uvicorn backend.app:app --reload
# open http://127.0.0.1:8000
```

## Train on real data (Kaggle)
| Dataset | Images | Setting | Use it for |
|---|---|---|---|
| **Merged lab+field (recommended)** `alinedobrovsky/plant-disease-classification-merged-dataset` | ~100K+, 38 classes | lab + real field | production model, works on phone photos |
| PlantVillage-38 `vipoooool/new-plant-diseases-dataset` | ~87K, 38 classes | lab, plain background | fastest baseline (95–99% val) |
| PlantDoc `nirmalsankalana/plantdoc-dataset` | 2,598, 27 classes | real farms/fields | fine-tune + generalization check |

```powershell
pip install kaggle   # add API key from kaggle.com → Account → ~/.kaggle/kaggle.json
kaggle datasets download -d alinedobrovsky/plant-disease-classification-merged-dataset -p ./data_raw --unzip
# or one-step: python -m backend.train --from-kaggle merged --epochs 15
pip install -r train-requirements.txt
# GPU (Windows/NVIDIA): pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m backend.train --data ./data_raw --epochs 15 --img 224 --batch 32
```
Training runs on **CUDA GPU when available** (model compute, mixed precision) with **multi-worker CPU data loading** - both processors work together; otherwise it falls back to CPU. Outputs land in `model/`: `plant_disease_mobilenet.pt`, refreshed `labels.json` / `metrics.json`, `training_curves.png`, `confusion_matrix.png`. Restart the app - the API picks up the trained weights automatically (`engine: mobilenetv2-trained (pytorch, cuda)` in `/api/health`).

## Pages
- `/` - homepage: hero, leaf diagnosis (upload + predict), accuracy banner, disease gallery, how-it-works.
- `/about` - model card: test/val accuracy, per-class P/R/F1 charts + table, datasets, training recipe, deploy notes.

## Deploy free on Render
1. Push this folder to GitHub.
2. Render → New → Web Service → select repo (it auto-detects `render.yaml`).
3. Build: `pip install -r requirements.txt` · Start: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
4. Open the Render URL - frontend + API on one domain, no CORS setup.

## API
- `GET /api/health` → `{status, engine, loaded, num_classes}`
- `GET /api/classes` → all classes with crop/disease/cause/treatment
- `GET /api/metrics` → test/val accuracy, macro & weighted avg, per-class P/R/F1
- `POST /api/predict` (multipart `file`) → `{prediction, confidence, top3, info, features}`

> Demo-grade predictions - always confirm with an agronomist before spraying.
