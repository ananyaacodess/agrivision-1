"""Real-life dataset catalogue (single source of truth).

Rendered dynamically on the About page via GET /api/datasets and used by
backend/train.py --from-kaggle presets.
"""

DATASETS = [
    {
        "id": "merged",
        "name": "Plant Disease Merged Dataset (lab + field)",
        "kaggle": "alinedobrovsky/plant-disease-classification-merged-dataset",
        "images": "~100K+",
        "classes": "38 (food crops)",
        "setting": "Laboratory + real field photos",
        "pros": "Best generalization to phone photos; large volume; cleaned (no watermarks, no single-condition classes).",
        "cons": "Bigger download and longer training.",
        "recommended": True,
    },
    {
        "id": "plantvillage38",
        "name": "New Plant Diseases Dataset (PlantVillage, augmented)",
        "kaggle": "vipoooool/new-plant-diseases-dataset",
        "images": "~87K RGB",
        "classes": "38",
        "setting": "Laboratory, plain backgrounds (pre-split train/valid/test folders)",
        "pros": "Huge and clean; fastest path to 95–99% validation accuracy; ready-made splits.",
        "cons": "Lab conditions only - accuracy drops on cluttered real-world photos.",
        "recommended": False,
    },
    {
        "id": "plantdoc",
        "name": "PlantDoc (real field photos)",
        "kaggle": "nirmalsankalana/plantdoc-dataset",
        "images": "2,598",
        "classes": "27 across 13 species",
        "setting": "Real farms/fields, cluttered backgrounds (object-detection style)",
        "pros": "True real-life distribution; best test of generalization (SOTA ~73–77%).",
        "cons": "Small and noisy; use for fine-tuning/evaluation rather than training from scratch.",
        "recommended": False,
    },
]

KAGGLE_SETUP = """# 1. Install the Kaggle CLI and add your API key (kaggle.com → Account → Create API token → save as ~/.kaggle/kaggle.json)
pip install kaggle

# 2. Download the RECOMMENDED merged lab+field dataset (~a few GB)
kaggle datasets download -d alinedobrovsky/plant-disease-classification-merged-dataset -p ./data_raw --unzip

# Alternatives:
# kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p ./data_raw --unzip
# kaggle datasets download -d nirmalsankalana/plantdoc-dataset -p ./data_raw --unzip

# 3. Train (auto-detects train/valid splits or flat class folders)
python -m backend.train --data ./data_raw --epochs 15 --img 224 --batch 32"""
