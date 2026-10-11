"""Static agronomy reference: cause + treatment per class."""

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": "Apple", "disease": "Apple Scab",
        "cause": "Fungus Venturia inaequalis. Olive-green/brown scabby spots on leaves and fruit, favored by wet spring weather.",
        "treatment": "Prune for airflow, remove fallen leaves. Fungicide with captan/myclobutanil. Resistant varieties for replanting.",
        "severity": "Moderate",
    },
    "Apple___Black_rot": {
        "crop": "Apple", "disease": "Black Rot",
        "cause": "Fungus Botryosphaeria obtusa. Purple-edged brown leaf spots ('frog-eye'), fruit rot and cankers.",
        "treatment": "Remove mummified fruit + cankers. Copper or captan spray. Improve drainage and spacing.",
        "severity": "High",
    },
    "Apple___healthy": {
        "crop": "Apple", "disease": "Healthy",
        "cause": "No disease detected. Leaf looks vigorous.",
        "treatment": "Maintain schedule: balanced NPK, mulch, monitor weekly. No spray needed.",
        "severity": "None",
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Maize", "disease": "Common Rust",
        "cause": "Fungus Puccinia sorghi. Cinnamon-brown pustules on both leaf surfaces.",
        "treatment": "Resistant hybrids next season. Azoxystrobin/propiconazole if spreading fast. Avoid overhead irrigation.",
        "severity": "Moderate",
    },
    "Corn_(maize)___healthy": {
        "crop": "Maize", "disease": "Healthy",
        "cause": "No disease detected.",
        "treatment": "Continue nitrogen top-dress schedule, scout for fall armyworm.",
        "severity": "None",
    },
    "Grape___Black_rot": {
        "crop": "Grape", "disease": "Black Rot",
        "cause": "Fungus Guignardia bidwellii. Circular brown leaf spots with black dots; berries shrivel to black mummies.",
        "treatment": "Destroy mummies, canopy thinning. Mancozeb/myclobutanil spray pre-bloom. Keep fruit zone dry.",
        "severity": "High",
    },
    "Grape___healthy": {
        "crop": "Grape", "disease": "Healthy",
        "cause": "No disease detected.",
        "treatment": "Maintain trellis pruning, sulfur spray as preventive in humid weeks.",
        "severity": "None",
    },
    "Potato___Early_blight": {
        "crop": "Potato", "disease": "Early Blight",
        "cause": "Fungus Alternaria solani. Concentric 'target' rings, yellowing from older leaves upward.",
        "treatment": "Chlorothalonil/mancozeb spray, crop rotation 2-3 yrs, remove lower infected leaves.",
        "severity": "Moderate",
    },
    "Potato___Late_blight": {
        "crop": "Potato", "disease": "Late Blight",
        "cause": "Oomycete Phytophthora infestans. Water-soaked pale lesions + white mildew underside; spreads explosively in cool wet weather.",
        "treatment": "Act fast: destroy infected foliage, copper/metalaxyl spray, stop irrigation. Do not compost infected material.",
        "severity": "Critical",
    },
    "Potato___healthy": {
        "crop": "Potato", "disease": "Healthy",
        "cause": "No disease detected.",
        "treatment": "Hill soil around stems, even watering, watch for Colorado beetle.",
        "severity": "None",
    },
    "Tomato___Early_blight": {
        "crop": "Tomato", "disease": "Early Blight",
        "cause": "Alternaria solani. Dark target spots starting on lower leaves.",
        "treatment": "Mulch to stop soil splash, stake plants, chlorothalonil or neem on schedule.",
        "severity": "Moderate",
    },
    "Tomato___Late_blight": {
        "crop": "Tomato", "disease": "Late Blight",
        "cause": "Phytophthora infestans. Greasy brown patches, white fuzz under leaf; fruit turns brown.",
        "treatment": "Remove + bag infected parts immediately, copper fungicide, space plants for airflow.",
        "severity": "Critical",
    },
    "Tomato___healthy": {
        "crop": "Tomato", "disease": "Healthy",
        "cause": "No disease detected.",
        "treatment": "Water at base, calcium to prevent blossom-end rot, weekly scouting.",
        "severity": "None",
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper", "disease": "Bacterial Spot",
        "cause": "Xanthomonas bacteria. Small water-soaked spots turning brown, leaf drop, fruit spots.",
        "treatment": "Copper + mancozeb spray, certified seed, drip irrigation, rotate out of solanaceae 2 yrs.",
        "severity": "High",
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper", "disease": "Healthy",
        "cause": "No disease detected.",
        "treatment": "Balanced feed, avoid leaf wetness, monitor for aphids/whitefly.",
        "severity": "None",
    },
}


def get_info(label: str) -> dict:
    """Return cause/treatment for a class, with a generic fallback for
    classes added by later training runs (e.g. the full 38-class set)."""
    if label in DISEASE_INFO:
        return DISEASE_INFO[label]
    parts = label.split("___")
    crop = parts[0].replace("_", " ").replace(",", "").strip().title()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    healthy = "healthy" in disease.lower()
    return {
        "crop": crop or "Unknown",
        "disease": disease,
        "cause": "No disease detected. Leaf looks vigorous." if healthy
        else f"Symptoms consistent with {disease} on {crop}. Confirm visually against reference images.",
        "treatment": "Maintain schedule: balanced feed, mulch, monitor weekly. No spray needed." if healthy
        else "Isolate affected plants, remove badly infected leaves (bag, do not compost). "
             "Apply a copper or broad-spectrum fungicide per label, improve airflow, "
             "avoid overhead watering, and consult a local agronomist for exact dosage.",
        "severity": "None" if healthy else "Moderate",
    }

