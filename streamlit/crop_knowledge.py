"""Crop configuration and tomato disease knowledge for AgriGuard AI."""

# Active crop keeps the current tomato model as the default while making it easy
# to add more crop entries later.
ACTIVE_CROP = "tomato"

TOMATO_CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
]

TOMATO_DISPLAY_NAMES = {
    "Tomato_Bacterial_spot": "Bacterial Spot",
    "Tomato_Early_blight": "Early Blight",
    "Tomato_Late_blight": "Late Blight",
    "Tomato_Leaf_Mold": "Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider Mites",
    "Tomato_Target_Spot": "Target Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato_Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Healthy",
}

TOMATO_SUSTAINABILITY_SCORES = {
    "Tomato_healthy": 100,
    "Tomato_Early_blight": 75,
    "Tomato_Late_blight": 40,
    "Tomato_Bacterial_spot": 65,
    "Tomato_Leaf_Mold": 60,
    "Tomato_Septoria_leaf_spot": 55,
    "Tomato_Spider_mites_Two_spotted_spider_mite": 50,
    "Tomato_Target_Spot": 55,
    "Tomato_Tomato_YellowLeaf_Curl_Virus": 35,
    "Tomato_Tomato_mosaic_virus": 35,
}

CROP_CONFIGS = {
    "tomato": {
        "crop_name": "Tomato",
        "model_name": "MobileNetV2",
        "model_path": "model/tomato_disease_model.keras",
        "image_size": (224, 224),
        "class_names": TOMATO_CLASS_NAMES,
        "display_names": TOMATO_DISPLAY_NAMES,
        "sustainability_scores": TOMATO_SUSTAINABILITY_SCORES,
    }
}


def get_crop_config(crop_key=ACTIVE_CROP):
    """Return crop configuration with tomato as the safe default."""
    return CROP_CONFIGS.get(crop_key, CROP_CONFIGS[ACTIVE_CROP])


def get_display_name(class_name, crop_key=ACTIVE_CROP):
    """Return a farmer-friendly disease name."""
    config = get_crop_config(crop_key)
    return config["display_names"].get(class_name, class_name.replace("_", " "))


def get_sustainability_score(class_name, crop_key=ACTIVE_CROP):
    """Return the sustainability score for a disease class."""
    config = get_crop_config(crop_key)
    return config["sustainability_scores"].get(class_name, 60)


crop_knowledge = {
    "Tomato_Bacterial_spot": {
        "cause": """
Caused by Xanthomonas bacteria.
Spread through infected seeds, splashing water, contaminated tools, and crop debris.
""",
        "symptoms": """
Small dark water-soaked leaf spots.
Yellow halos around lesions.
Scabby fruit blemishes and premature leaf drop.
""",
        "medication": """
Copper bactericides.
Streptomycin where locally approved.
Use certified disease-free seeds and transplants.
""",
    },
    "Tomato_Early_blight": {
        "cause": """
Caused by Alternaria solani fungus.
Usually develops during warm, humid weather and spreads from infected crop residue.
""",
        "symptoms": """
Brown concentric spots on older leaves.
Yellowing around lesions.
Leaf drop and reduced plant vigor.
""",
        "medication": """
Mancozeb.
Chlorothalonil.
Copper-based fungicides.
""",
    },
    "Tomato_Late_blight": {
        "cause": """
Caused by Phytophthora infestans.
Spreads rapidly during cool and humid conditions, especially after rain or heavy dew.
""",
        "symptoms": """
Dark brown lesions.
Water-soaked leaf spots.
Rapid wilting, stem lesions, and fruit rot.
""",
        "medication": """
Mancozeb.
Copper oxychloride.
Chlorothalonil or other approved late blight fungicides.
""",
    },
    "Tomato_Leaf_Mold": {
        "cause": """
Caused by the fungus Passalora fulva.
Common in greenhouses or dense plantings with high humidity and poor airflow.
""",
        "symptoms": """
Yellow patches on upper leaf surfaces.
Olive-green to gray mold under leaves.
Leaf curling, drying, and defoliation in severe cases.
""",
        "medication": """
Improve ventilation.
Use sulfur, copper, or recommended fungicides when needed.
Remove heavily infected leaves.
""",
    },
    "Tomato_Septoria_leaf_spot": {
        "cause": """
Caused by Septoria lycopersici fungus.
Survives on plant debris and spreads by rain splash, irrigation splash, and tools.
""",
        "symptoms": """
Many small circular spots with gray centers.
Dark borders and tiny black fruiting bodies in lesions.
Lower leaves yellow, dry, and fall.
""",
        "medication": """
Chlorothalonil.
Mancozeb.
Copper-based fungicides as per local recommendations.
""",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "cause": """
Caused by two-spotted spider mite feeding.
Hot, dry conditions and dusty plants increase infestation risk.
""",
        "symptoms": """
Fine yellow stippling on leaves.
Webbing under leaves.
Bronzing, leaf drying, and plant stress under heavy infestation.
""",
        "medication": """
Spray water under leaves.
Use neem oil or insecticidal soap.
Use approved miticides only when infestation is severe.
""",
    },
    "Tomato_Target_Spot": {
        "cause": """
Caused by Corynespora cassiicola fungus.
Favored by warm, wet weather and dense plant canopy.
""",
        "symptoms": """
Brown circular lesions with target-like rings.
Spots may merge into larger dead areas.
Severe infection causes leaf drop and fruit lesions.
""",
        "medication": """
Chlorothalonil.
Mancozeb.
Use recommended fungicide rotation to reduce resistance risk.
""",
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "cause": """
Caused by Tomato yellow leaf curl virus.
Spread mainly by whiteflies and infected transplants.
""",
        "symptoms": """
Upward leaf curling.
Yellow leaf margins.
Stunted plants, flower drop, and reduced fruit set.
""",
        "medication": """
No curative chemical treatment for infected plants.
Control whiteflies.
Remove infected plants and use resistant varieties.
""",
    },
    "Tomato_Tomato_mosaic_virus": {
        "cause": """
Caused by Tomato mosaic virus.
Spread mechanically through hands, tools, plant contact, and infected seed.
""",
        "symptoms": """
Mottled light and dark green leaves.
Leaf distortion and stunted growth.
Uneven fruit ripening in severe cases.
""",
        "medication": """
No curative treatment for infected plants.
Remove infected plants.
Disinfect tools and avoid tobacco contamination.
""",
    },
    "Tomato_healthy": {
        "cause": "No disease detected.",
        "symptoms": """
Healthy green leaves.
Normal growth and no visible disease symptoms.
""",
        "medication": """
No medication required.
Continue monitoring and maintain balanced crop nutrition.
""",
    },
}
