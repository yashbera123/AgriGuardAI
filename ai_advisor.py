"""
AI Agriculture Advisor module for AgriGuard AI.

This module provides:
    - Disease explanation generator (farmer-friendly language)
    - Prevention recommendation generator
    - Sustainable farming recommendation generator
    - Long-term crop protection recommendation generator
    - Chatbot response generator with smart disease context

IBM BOB Integration:
    This module is designed to be IBM BOB-ready. Search for the comments marked
    "# IBM BOB INTEGRATION POINT" to find exact locations where you can swap the
    local knowledge-base logic for live IBM BOB API calls.

Safety Disclaimer:
    Every advisory and chatbot response automatically appends the following:
    "Consult local agricultural experts before applying chemicals or
     changing farming practices."
"""

# =============================================================================
# IBM BOB INTEGRATION POINT — Future import block
# When IBM BOB is available, uncomment and configure:
#
#   import requests  # or the official IBM BOB SDK
#   IBM_BOB_API_URL = "https://your-ibm-bob-endpoint.ibm.com/api/v1/chat"
#   IBM_BOB_API_KEY = os.environ.get("IBM_BOB_API_KEY", "")
# =============================================================================

# ---------------------------------------------------------------------------
# Safety Disclaimer (appended to every advisory and chatbot response)
# ---------------------------------------------------------------------------
SAFETY_DISCLAIMER = (
    " Consult local agricultural experts before applying chemicals or "
    "changing farming practices."
)

# ---------------------------------------------------------------------------
# Per-disease advisory knowledge base
# ---------------------------------------------------------------------------
# Each disease entry contains five advisory fields:
#   farmer_explanation  — plain-language what/why for a farmer
#   prevention          — actionable prevention steps
#   sustainability      — eco-friendly, low-input approaches
#   treatment_strategy  — step-by-step treatment playbook
#   long_term_advice    — season-over-season crop protection guidance
# ---------------------------------------------------------------------------

_ADVISORY_KB = {
    "Tomato_Bacterial_spot": {
        "farmer_explanation": (
            "Bacterial Spot is caused by tiny bacteria that hide in seeds and "
            "splash onto leaves with water. It looks like small dark spots with "
            "yellow edges on your leaves and scabby marks on fruits. It spreads "
            "very fast in warm and rainy weather, so quick action saves the crop."
        ),
        "prevention": (
            "1. Always buy certified disease-free seeds and seedlings.\n"
            "2. Avoid overhead sprinklers — water at the base of the plant.\n"
            "3. Remove and destroy infected leaves and plant debris immediately.\n"
            "4. Rotate tomatoes away from the same field for at least 2 seasons.\n"
            "5. Clean and disinfect all farming tools between uses."
        ),
        "sustainability": (
            "• Use drip irrigation to keep leaves dry and reduce spread.\n"
            "• Apply compost to build soil health and plant immunity.\n"
            "• Encourage crop diversity — mixed planting reduces disease pressure.\n"
            "• Copper-based sprays are among the lower-impact bactericides; use "
            "only when necessary and rotate to avoid resistance.\n"
            "• Collect and destroy crop residue after harvest instead of leaving it."
        ),
        "treatment_strategy": (
            "Step 1: Inspect all plants and mark infected ones.\n"
            "Step 2: Remove and bag infected leaves — do not compost them.\n"
            "Step 3: Avoid working with plants when they are wet.\n"
            "Step 4: Apply copper-based bactericide as per label instructions.\n"
            "Step 5: Re-inspect every 5–7 days and repeat treatment if new spots appear.\n"
            "Step 6: Improve field drainage to reduce standing water near roots."
        ),
        "long_term_advice": (
            "• Rotate crops — do not grow tomatoes or peppers in the same plot for "
            "at least 2 consecutive seasons.\n"
            "• Select resistant or tolerant tomato varieties for future plantings.\n"
            "• Build a seed-health routine: only use certified or hot-water treated seed.\n"
            "• Keep a field diary to track disease outbreaks by date and location.\n"
            "• Train farm workers on sanitation practices (clean hands and tools)."
        ),
    },

    "Tomato_Early_blight": {
        "farmer_explanation": (
            "Early Blight is caused by a common soil fungus called Alternaria. "
            "It usually attacks older, lower leaves first, creating brown spots with "
            "rings inside them — like a target. It weakens the plant slowly and can "
            "spread to the fruit. Warm, humid weather speeds up its spread."
        ),
        "prevention": (
            "1. Lay mulch around plant bases to stop soil from splashing onto leaves.\n"
            "2. Space plants widely enough for good air circulation.\n"
            "3. Rotate crops — avoid planting tomatoes where infected plants grew last season.\n"
            "4. Water in the morning so leaves dry out during the day.\n"
            "5. Remove lower leaves that touch the soil regularly."
        ),
        "sustainability": (
            "• Mulching with straw or dry leaves reduces fungal splash and retains moisture.\n"
            "• Neem oil spray is an organic option for early-stage management.\n"
            "• Compost tea can improve leaf surface microbiome and resist fungal attack.\n"
            "• Chemical fungicides should be a last resort — prioritise cultural controls first.\n"
            "• Healthy, well-fed plants are more resistant; balance nitrogen and potassium."
        ),
        "treatment_strategy": (
            "Step 1: Prune and remove all infected lower leaves — bag and destroy them.\n"
            "Step 2: Apply an approved fungicide (Mancozeb or Chlorothalonil) to leaves.\n"
            "Step 3: Avoid overhead watering during treatment period.\n"
            "Step 4: Re-apply fungicide every 7–10 days in humid weather.\n"
            "Step 5: Monitor plant recovery and new growth.\n"
            "Step 6: Add potassium-rich fertiliser to support plant recovery."
        ),
        "long_term_advice": (
            "• Do not grow tomatoes, potatoes, or eggplant on the same plot consecutively.\n"
            "• Choose early blight-resistant tomato varieties for future seasons.\n"
            "• Deep plough after harvest to bury infected crop residue underground.\n"
            "• Maintain a strong plant nutrition programme — stressed plants get sick faster.\n"
            "• Scout the lower canopy weekly during warm, humid periods."
        ),
    },

    "Tomato_Late_blight": {
        "farmer_explanation": (
            "Late Blight is one of the most dangerous tomato diseases — it was "
            "responsible for historical famines. It is caused by a water mould that "
            "spreads incredibly fast in cool, wet weather. Dark brown lesions appear "
            "on leaves, stems, and fruit, and the entire plant can collapse in days. "
            "Act immediately when you see it."
        ),
        "prevention": (
            "1. Monitor forecasts — extra vigilance during cool, rainy spells.\n"
            "2. Avoid prolonged leaf wetness — do not water in the evening.\n"
            "3. Remove and destroy volunteer tomato plants and infected debris.\n"
            "4. Use resistant varieties in areas with a history of Late Blight.\n"
            "5. Apply preventive copper or Mancozeb sprays before humid periods."
        ),
        "sustainability": (
            "• Preventive spraying in high-risk weather is more sustainable than "
            "fighting a full outbreak (less chemical use overall).\n"
            "• Improve field drainage — waterlogged soils accelerate spread.\n"
            "• Avoid over-dense planting; airflow is a natural fungal barrier.\n"
            "• Copper fungicides are among the lower-impact options but can build "
            "up in soil — rotate with other approved products.\n"
            "• Remove all infected plant material from the field immediately."
        ),
        "treatment_strategy": (
            "Step 1: Remove infected plants or plant parts IMMEDIATELY — bag them.\n"
            "Step 2: Do NOT compost infected material — burn or bury it far from the field.\n"
            "Step 3: Apply an approved fungicide (Mancozeb, Copper Oxychloride) to all plants.\n"
            "Step 4: Increase plant spacing and improve field drainage if possible.\n"
            "Step 5: Reapply fungicide every 5–7 days during wet conditions.\n"
            "Step 6: Alert neighbouring farmers — Late Blight spreads long distances by wind."
        ),
        "long_term_advice": (
            "• Plant Late Blight-resistant tomato varieties in future seasons.\n"
            "• Implement a crop rotation of at least 3 years for tomatoes and potatoes.\n"
            "• Set up a local disease alert system with neighbouring farmers.\n"
            "• Never save seeds from infected fruits.\n"
            "• Consider raised beds or greenhouses if Late Blight is recurrent in your area."
        ),
    },

    "Tomato_Leaf_Mold": {
        "farmer_explanation": (
            "Leaf Mold is a fungal disease that loves greenhouses and dense plantings "
            "with high humidity. You will notice yellow patches on top of leaves and "
            "an olive-green fuzzy mold underneath. It rarely kills plants directly but "
            "reduces yield significantly if left unchecked. Better airflow is the most "
            "powerful weapon against it."
        ),
        "prevention": (
            "1. Keep greenhouse humidity below 85% — use vents or fans.\n"
            "2. Space plants well apart to allow air movement between them.\n"
            "3. Prune lower and inner leaves to open up the canopy.\n"
            "4. Avoid watering late in the day when humidity rises at night.\n"
            "5. Use resistant tomato varieties whenever available."
        ),
        "sustainability": (
            "• Physical airflow management (pruning, venting) is the most sustainable approach.\n"
            "• Drip irrigation keeps leaves dry — a key prevention strategy.\n"
            "• Sulfur-based fungicides are lower-impact and effective for Leaf Mold.\n"
            "• Regularly removing infected leaves reduces spore load naturally.\n"
            "• Avoid over-fertilising with nitrogen, which promotes dense, humid canopies."
        ),
        "treatment_strategy": (
            "Step 1: Remove all heavily infected leaves and bag them for disposal.\n"
            "Step 2: Increase ventilation — open vents, add fans in greenhouses.\n"
            "Step 3: Reduce irrigation frequency to lower ambient humidity.\n"
            "Step 4: Apply sulfur, copper, or recommended fungicide to remaining foliage.\n"
            "Step 5: Re-inspect every 7 days and repeat treatment if needed.\n"
            "Step 6: Adjust plant spacing for the next growing season."
        ),
        "long_term_advice": (
            "• Invest in proper greenhouse ventilation infrastructure.\n"
            "• Select Leaf Mold-resistant varieties (look for Cf resistance genes).\n"
            "• Practise seasonal deep cleaning of greenhouse surfaces.\n"
            "• Track humidity and temperature daily — address spikes quickly.\n"
            "• Plan planting density to allow at least 50 cm between plants."
        ),
    },

    "Tomato_Septoria_leaf_spot": {
        "farmer_explanation": (
            "Septoria Leaf Spot is caused by a fungus that hides in old plant debris. "
            "It causes many small circular spots with pale grey centres and dark "
            "borders on the lower leaves first. Heavy rain or irrigation splash spreads "
            "it up the plant. It rarely kills plants outright but causes heavy leaf "
            "drop that weakens the crop and reduces yield."
        ),
        "prevention": (
            "1. Lay mulch around plant bases to stop soil splash.\n"
            "2. Water at the soil level — avoid wetting leaves.\n"
            "3. Stake plants to keep leaves elevated off the ground.\n"
            "4. Remove all infected plant debris at the end of each season.\n"
            "5. Rotate crops — avoid tomatoes in the same soil for 2+ seasons."
        ),
        "sustainability": (
            "• Staking and mulching are chemical-free ways to dramatically cut disease risk.\n"
            "• Organic copper fungicides offer moderate control with less environmental impact.\n"
            "• Sanitation (removing crop debris) eliminates fungal spores from the field.\n"
            "• Cover crops in rotation can suppress soil-borne pathogens naturally.\n"
            "• Intercropping with basil or marigolds may reduce fungal pressure."
        ),
        "treatment_strategy": (
            "Step 1: Remove and destroy infected lower leaves immediately.\n"
            "Step 2: Apply mulch around plants to prevent further soil splash.\n"
            "Step 3: Apply Chlorothalonil or Mancozeb fungicide to all plants.\n"
            "Step 4: Reapply every 7–10 days during warm, wet conditions.\n"
            "Step 5: Water only at soil level until infection is controlled.\n"
            "Step 6: Monitor for new spots on upper leaves weekly."
        ),
        "long_term_advice": (
            "• Practise 2–3 year rotation cycles away from tomatoes.\n"
            "• Till and bury crop debris deeply after harvest to destroy overwintering spores.\n"
            "• Use certified seed from clean sources each season.\n"
            "• Select tomato varieties with improved tolerance to foliar diseases.\n"
            "• Train farm workers to recognise early symptoms for rapid response."
        ),
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "farmer_explanation": (
            "Spider Mites are tiny pests (barely visible to the naked eye) that suck "
            "sap from the underside of leaves, leaving a fine yellow stippling pattern. "
            "In heavy infestations, you will see fine silky webbing under leaves. They "
            "thrive in hot, dry, dusty conditions and can reproduce very rapidly — a "
            "small colony can become a major problem in just one week."
        ),
        "prevention": (
            "1. Keep plants well-watered — stressed plants attract mites.\n"
            "2. Reduce dust around the crop by wetting pathways.\n"
            "3. Inspect the underside of leaves weekly during hot, dry weather.\n"
            "4. Introduce natural predators (ladybirds, predatory mites) if available.\n"
            "5. Avoid excessive nitrogen fertilisation, which produces tender, mite-friendly leaves."
        ),
        "sustainability": (
            "• Spray strong jets of water under leaves — this dislodges mites cheaply.\n"
            "• Neem oil and insecticidal soap are low-toxicity options effective against mites.\n"
            "• Encourage beneficial insects such as predatory mites (Phytoseiidae family).\n"
            "• Chemical miticides should be a last resort — overuse causes rapid resistance.\n"
            "• Intercropping with dill or coriander attracts natural predators."
        ),
        "treatment_strategy": (
            "Step 1: Use a strong water spray on the underside of all leaves.\n"
            "Step 2: Apply neem oil or insecticidal soap solution — coat leaves thoroughly.\n"
            "Step 3: Remove the most heavily infested leaves.\n"
            "Step 4: If population is severe, apply an approved miticide per label directions.\n"
            "Step 5: Rotate miticide class to prevent resistance — do not use the same product twice.\n"
            "Step 6: Repeat treatment every 5–7 days until mites are under control."
        ),
        "long_term_advice": (
            "• Build up a natural enemy population in and around the farm.\n"
            "• Avoid broad-spectrum pesticide sprays that kill beneficial insects.\n"
            "• Keep irrigation consistent — drought stress is the number-one mite trigger.\n"
            "• After the season, clean up crop debris where mite eggs overwinter.\n"
            "• Consider shade netting to reduce temperature spikes that encourage mite outbreaks."
        ),
    },

    "Tomato_Target_Spot": {
        "farmer_explanation": (
            "Target Spot is caused by a fungus that creates brown circular lesions "
            "with clear concentric rings inside — much like a target or bullseye. It "
            "affects leaves, stems, and sometimes fruit. Warm, wet conditions and "
            "dense plant canopies help it spread quickly. Timely fungicide application "
            "and improved airflow are key to managing it."
        ),
        "prevention": (
            "1. Maintain wide plant spacing for good air circulation.\n"
            "2. Avoid overhead watering — water at soil level.\n"
            "3. Prune lower, crowded leaves to open up the canopy.\n"
            "4. Remove infected crop residue promptly.\n"
            "5. Rotate fungicide products between applications to manage resistance."
        ),
        "sustainability": (
            "• Cultural controls (spacing, pruning) reduce the need for fungicides.\n"
            "• Drip irrigation keeps the foliar environment drier and less hospitable to fungi.\n"
            "• Apply organic mulch to prevent soil splash of spores onto lower leaves.\n"
            "• Copper-based fungicides have a relatively low environmental footprint.\n"
            "• Integrated pest management (IPM) combines cultural, biological, and chemical tools."
        ),
        "treatment_strategy": (
            "Step 1: Remove and destroy all infected foliage immediately.\n"
            "Step 2: Prune the canopy to improve airflow around plants.\n"
            "Step 3: Apply Chlorothalonil or Mancozeb fungicide as per label.\n"
            "Step 4: Rotate to a different fungicide class after 2 consecutive applications.\n"
            "Step 5: Re-inspect every 7 days and treat if new lesions appear.\n"
            "Step 6: Reduce overhead watering during the treatment period."
        ),
        "long_term_advice": (
            "• Plan field layout to ensure adequate row spacing for airflow.\n"
            "• Rotate crops to break the disease cycle in the soil.\n"
            "• Select varieties with improved foliar disease tolerance.\n"
            "• Implement a strict end-of-season field clean-up programme.\n"
            "• Document spray records to rotate fungicide classes effectively."
        ),
    },

    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "farmer_explanation": (
            "Yellow Leaf Curl Virus is a serious viral disease spread by tiny white "
            "insects called whiteflies. Infected plants curl their leaves upward, turn "
            "yellow at the edges, and stop producing fruit properly. There is no cure "
            "once a plant is infected — the focus must be on controlling whiteflies and "
            "removing infected plants before the virus spreads to healthy ones."
        ),
        "prevention": (
            "1. Use yellow sticky traps to monitor and catch whiteflies early.\n"
            "2. Install insect-proof netting in nurseries to protect seedlings.\n"
            "3. Remove weeds around the field that can harbour whiteflies.\n"
            "4. Buy seedlings only from certified, disease-free nurseries.\n"
            "5. Plant resistant tomato varieties wherever they are available."
        ),
        "sustainability": (
            "• Sticky traps are a chemical-free early warning and control tool.\n"
            "• Reflective mulches confuse and repel whiteflies without chemicals.\n"
            "• Neem oil spray can suppress whitefly populations with low environmental impact.\n"
            "• Removing infected plants immediately is the most sustainable disease control.\n"
            "• Intercropping with strongly aromatic plants may deter whitefly migration."
        ),
        "treatment_strategy": (
            "Step 1: Identify and immediately remove all plants showing symptoms.\n"
            "Step 2: Bag removed plants tightly and destroy them — do not compost.\n"
            "Step 3: Spray remaining healthy plants with neem oil or approved insecticide.\n"
            "Step 4: Place yellow sticky traps every 10 metres around the field perimeter.\n"
            "Step 5: Check traps and plants every 3–5 days for whitefly activity.\n"
            "Step 6: Maintain clean field edges by removing all weeds and volunteer plants."
        ),
        "long_term_advice": (
            "• Transition to virus-resistant tomato varieties — this is the most reliable defence.\n"
            "• Establish a whitefly monitoring programme year-round in high-risk areas.\n"
            "• Work with neighbouring farmers to coordinate whitefly control across fields.\n"
            "• Avoid overlapping tomato crops from season to season to break the disease cycle.\n"
            "• Report severe outbreaks to local agricultural extension officers."
        ),
    },

    "Tomato_Tomato_mosaic_virus": {
        "farmer_explanation": (
            "Tomato Mosaic Virus spreads through physical contact — on hands, tools, "
            "clothing, and even tobacco products. It creates a mottled pattern of light "
            "and dark green on leaves, distorts plant shape, and stunts growth. Unlike "
            "fungal diseases, there is no chemical cure — prevention and hygiene are "
            "everything."
        ),
        "prevention": (
            "1. Wash hands thoroughly before handling any plants.\n"
            "2. Sterilise pruning tools between each plant using bleach or alcohol.\n"
            "3. Use virus-free certified seed — never save seed from infected plants.\n"
            "4. Avoid smoking near or handling plants after using tobacco products.\n"
            "5. Remove and destroy infected plants at the first sign of symptoms."
        ),
        "sustainability": (
            "• Hand and tool hygiene is completely chemical-free and highly effective.\n"
            "• Removing infected plants early prevents exponential spread at zero cost.\n"
            "• Certified seed programmes reduce viral risk from the very start of the season.\n"
            "• Good general plant health (balanced nutrition, water) reduces symptom severity.\n"
            "• Biological controls are not available for viruses — prevention is the only tool."
        ),
        "treatment_strategy": (
            "Step 1: Identify and mark all infected plants.\n"
            "Step 2: Remove infected plants carefully — avoid touching healthy plants after.\n"
            "Step 3: Bag and destroy removed plants — do not compost or leave in the field.\n"
            "Step 4: Sterilise every tool used during removal with 10% bleach solution.\n"
            "Step 5: Wash hands and change gloves before continuing farm work.\n"
            "Step 6: Monitor remaining plants weekly for new symptoms."
        ),
        "long_term_advice": (
            "• Establish a strict tool sanitation policy as a farm-wide standard.\n"
            "• Train all farm workers on virus transmission and hygiene protocols.\n"
            "• Source seed only from accredited suppliers with virus-testing certificates.\n"
            "• Consider plant variety resistance ratings when selecting seed each season.\n"
            "• Keep a field map noting where viral infections appeared to guide future rotations."
        ),
    },

    "Tomato_healthy": {
        "farmer_explanation": (
            "Great news — your tomato plant appears healthy! No disease symptoms were "
            "detected. Healthy plants produce better yields, require fewer inputs, and "
            "support a more sustainable farming operation. Keep up your current practices "
            "and stay vigilant to catch any early signs of disease before they spread."
        ),
        "prevention": (
            "1. Scout all plants every 7–10 days — early detection is the best protection.\n"
            "2. Maintain consistent irrigation and balanced fertilisation.\n"
            "3. Keep field edges free of weeds and volunteer plants.\n"
            "4. Practise good tool and hand hygiene between plants.\n"
            "5. Keep a farm diary to track any changes in plant health over time."
        ),
        "sustainability": (
            "• Healthy crops naturally require fewer pesticides and fungicides.\n"
            "• Maintain soil organic matter with compost — healthy soil grows healthy plants.\n"
            "• Use drip irrigation to conserve water and keep leaves dry.\n"
            "• Practise crop rotation to protect soil health and prevent disease build-up.\n"
            "• Encourage beneficial insects by planting flowering species on field borders."
        ),
        "treatment_strategy": (
            "No treatment needed at this time.\n"
            "Continue: regular monitoring, balanced nutrition, and clean field practices.\n"
            "Apply preventive sprays only if weather forecasts indicate high disease risk.\n"
            "Maintain records of plant health observations for future reference."
        ),
        "long_term_advice": (
            "• Rotate crops every 2–3 seasons to prevent soil-borne disease build-up.\n"
            "• Invest in soil health — regular compost applications improve resilience.\n"
            "• Select improved varieties with disease resistance for future seasons.\n"
            "• Build relationships with local agricultural extension services.\n"
            "• Consider certification programmes (organic or GAP) to access premium markets."
        ),
    },
}

# Fallback advisory for unrecognised or novel diseases
_FALLBACK_ADVISORY = {
    "farmer_explanation": (
        "A disease condition has been detected in this plant. While specific "
        "information for this condition is not yet in the knowledge base, general "
        "plant disease management principles apply: isolate affected plants, improve "
        "ventilation and drainage, and consult a local agronomist for diagnosis."
    ),
    "prevention": (
        "1. Isolate infected plants from healthy ones immediately.\n"
        "2. Avoid spreading material from infected plants to healthy areas.\n"
        "3. Improve field hygiene — clean tools and wash hands regularly.\n"
        "4. Contact a local agricultural extension officer for accurate identification.\n"
        "5. Do not apply chemicals until the disease is properly identified."
    ),
    "sustainability": (
        "• Focus on cultural controls (spacing, sanitation, irrigation management) first.\n"
        "• Avoid broad-spectrum chemical applications until the problem is identified.\n"
        "• Healthy soil and balanced plant nutrition support natural disease resistance.\n"
        "• Document and photograph symptoms to help experts identify the problem."
    ),
    "treatment_strategy": (
        "Step 1: Photograph the symptoms from multiple angles for expert review.\n"
        "Step 2: Isolate affected plants and restrict movement within the field.\n"
        "Step 3: Contact your local agricultural extension service for field diagnosis.\n"
        "Step 4: Follow official recommendations once the disease is confirmed.\n"
        "Step 5: Monitor all remaining plants closely for spread."
    ),
    "long_term_advice": (
        "• Maintain accurate farm records to track disease history by field and season.\n"
        "• Build a relationship with your local agricultural extension service.\n"
        "• Invest in plant health monitoring tools and training.\n"
        "• Practise crop rotation and field sanitation as universal good practices."
    ),
}

# ---------------------------------------------------------------------------
# Chatbot response knowledge base (frequently asked questions)
# ---------------------------------------------------------------------------
_CHATBOT_KB = {
    # Disease-specific Q&A patterns: mapped to disease class names
    "Tomato_Late_blight": {
        "prevent": (
            "To prevent Late Blight on your tomato crop:\n\n"
            "1. Monitor weather forecasts — act before cool, wet conditions arrive.\n"
            "2. Apply preventive copper or Mancozeb spray during high-risk weather.\n"
            "3. Avoid evening watering — wet leaves overnight boost spread.\n"
            "4. Remove volunteer tomato plants from around your field.\n"
            "5. Destroy all infected crop debris — do not compost it.\n\n"
            "Late Blight is extremely fast-moving. Prevention is far more effective than cure."
        ),
        "fungicide": (
            "For Late Blight, the following fungicides are commonly recommended:\n\n"
            "• **Mancozeb** — broad-spectrum, widely available, preventive action\n"
            "• **Copper Oxychloride** — lower resistance risk, suitable for organic-leaning farms\n"
            "• **Chlorothalonil** — effective preventive contact fungicide\n"
            "• **Cymoxanil + Mancozeb** — combination product for systemic + contact action\n\n"
            "Always check local registration — available products vary by country.\n"
            "Rotate fungicide classes to prevent resistance developing."
        ),
        "treat": (
            "To treat an active Late Blight infection:\n\n"
            "1. Remove infected plants or plant parts IMMEDIATELY.\n"
            "2. Bag removed material and burn or bury it far from the field.\n"
            "3. Apply an approved systemic fungicide to all remaining plants.\n"
            "4. Repeat every 5–7 days while wet conditions continue.\n"
            "5. Alert neighbouring farmers — spores travel long distances by wind."
        ),
    },
    "Tomato_Early_blight": {
        "prevent": (
            "To prevent Early Blight:\n\n"
            "1. Lay mulch around plant bases to stop soil splash.\n"
            "2. Space plants to allow good airflow between them.\n"
            "3. Remove lower leaves that touch the soil.\n"
            "4. Water in the morning at soil level — keep leaves dry.\n"
            "5. Rotate tomatoes away from the same field each season."
        ),
        "fungicide": (
            "For Early Blight, common fungicide options include:\n\n"
            "• **Mancozeb** — reliable, preventive contact fungicide\n"
            "• **Chlorothalonil** — effective against Alternaria species\n"
            "• **Copper-based fungicides** — suitable for organic programmes\n"
            "• **Azoxystrobin** — systemic option for active infections\n\n"
            "Apply preventively before symptoms appear during warm, humid weather."
        ),
    },
    "Tomato_Bacterial_spot": {
        "prevent": (
            "To prevent Bacterial Spot:\n\n"
            "1. Use certified disease-free seeds and transplants only.\n"
            "2. Avoid overhead irrigation — water at the base.\n"
            "3. Remove and destroy infected leaves immediately.\n"
            "4. Disinfect tools regularly with bleach or alcohol solution.\n"
            "5. Rotate crops for at least 2 seasons."
        ),
        "fungicide": (
            "Bacterial Spot is caused by bacteria, not fungi, so bactericides are used:\n\n"
            "• **Copper bactericides** — most widely used and available\n"
            "• **Streptomycin** — where locally approved and registered\n"
            "• Combined copper + Mancozeb products for dual-action protection\n\n"
            "Bactericides are most effective as preventives before infection occurs."
        ),
    },
}

# ---------------------------------------------------------------------------
# General farming knowledge for chatbot (disease-independent topics)
# ---------------------------------------------------------------------------
_GENERAL_KB = {
    "sustainable": (
        "Sustainable farming practices for tomatoes include:\n\n"
        " **Soil Health**\n"
        "• Add organic compost to improve soil structure and microbial life.\n"
        "• Use cover crops between seasons to fix nitrogen and prevent erosion.\n"
        "• Minimise tillage to protect soil structure.\n\n"
        " **Water Management**\n"
        "• Use drip irrigation to reduce water use and keep leaves dry.\n"
        "• Collect and use rainwater where possible.\n"
        "• Mulch around plants to retain soil moisture.\n\n"
        " **Integrated Pest Management (IPM)**\n"
        "• Monitor weekly rather than spraying on a fixed schedule.\n"
        "• Encourage beneficial insects with flowering border plants.\n"
        "• Use chemical sprays only when pest thresholds are exceeded.\n\n"
        " **Resource Efficiency**\n"
        "• Rotate crops every 2–3 seasons to break disease and pest cycles.\n"
        "• Use certified seeds to reduce disease pressure from the start.\n"
        "• Record yield and input data each season to optimise future decisions."
    ),
    "crop health": (
        "To improve and maintain tomato crop health:\n\n"
        "1. **Soil Test First** — know your soil pH and nutrient levels before planting.\n"
        "2. **Balanced Nutrition** — tomatoes need adequate nitrogen, phosphorus, potassium, and calcium.\n"
        "3. **Consistent Watering** — irregular watering causes blossom end rot and stress.\n"
        "4. **Proper Spacing** — gives plants light, air, and room to grow.\n"
        "5. **Scout Regularly** — inspect every plant every 7–10 days for early signs of disease.\n"
        "6. **Remove Stress Quickly** — address any disease, pest, or nutrition issue early.\n"
        "7. **Good Genetics** — choose varieties suited to your local climate and disease pressures."
    ),
    "fungicide": (
        "General fungicide guidance for tomato diseases:\n\n"
        "**Contact Fungicides** (preventive, stay on leaf surface):\n"
        "• Mancozeb, Chlorothalonil, Copper-based products\n\n"
        "**Systemic Fungicides** (absorbed into the plant, curative):\n"
        "• Azoxystrobin, Tebuconazole, Propiconazole\n\n"
        "**Key Principles:**\n"
        "• Rotate between different fungicide classes to prevent resistance.\n"
        "• Always read and follow label instructions and safety precautions.\n"
        "• Apply at the correct timing — preventive is always better than curative.\n"
        "• Check local product registrations — approved products vary by country."
    ),
    "ipm": (
        "Integrated Pest Management (IPM) for tomatoes:\n\n"
        "IPM combines multiple approaches to manage pests and diseases effectively "
        "while minimising chemical use and environmental impact.\n\n"
        "**4 Core Pillars of IPM:**\n"
        "1. **Monitoring** — scout fields regularly, set action thresholds.\n"
        "2. **Cultural Controls** — rotation, spacing, sanitation, resistant varieties.\n"
        "3. **Biological Controls** — beneficial insects, biopesticides.\n"
        "4. **Chemical Controls** — use as a last resort, rotate products.\n\n"
        "IPM reduces costs, delays resistance, and builds long-term farm resilience."
    ),
    "organic": (
        "Organic pest and disease management for tomatoes:\n\n"
        "**Approved Organic Inputs:**\n"
        "• Copper-based fungicides and bactericides (limited application per season)\n"
        "• Sulfur-based fungicides for foliar diseases\n"
        "• Neem oil for insects and some fungal pathogens\n"
        "• Insecticidal soap for soft-bodied insects like aphids and mites\n"
        "• Bacillus subtilis (biofungicide) for early disease control\n\n"
        "**Cultural Practices:**\n"
        "• Compost-enriched soil suppresses many soil-borne pathogens.\n"
        "• Diversity planting reduces pest build-up.\n"
        "• Physical barriers (netting, row covers) protect against insects."
    ),
    "irrigation": (
        "Smart irrigation practices for healthy tomato crops:\n\n"
        "• **Drip irrigation** is best — delivers water to roots, keeps leaves dry.\n"
        "• Water in the **morning** so any splashed leaves dry during the day.\n"
        "• **Avoid evening watering** — wet leaves overnight promote fungal diseases.\n"
        "• Keep irrigation **consistent** — irregular watering causes blossom end rot.\n"
        "• Use **mulch** to retain soil moisture and reduce watering frequency.\n"
        "• Monitor soil moisture with a probe or simple finger test before watering."
    ),
    "rotation": (
        "Crop rotation recommendations for tomatoes:\n\n"
        "• Avoid planting tomatoes, peppers, potatoes, or eggplant in the same field "
        "for at least **2–3 consecutive seasons**.\n"
        "• Good rotation crops include legumes (beans, peas), leafy greens, and cereals.\n"
        "• Legumes fix nitrogen and improve soil health for the next crop.\n"
        "• Rotation breaks the life cycle of soil-borne diseases like Early Blight, "
        "Septoria, and Fusarium.\n"
        "• Keep a simple field map to track what was grown where each season."
    ),
    "hello": (
        "Hello! I'm AgriGuard AI — your intelligent agriculture advisor. \n\n"
        "You can ask me about:\n"
        "• **Disease management** — prevention, treatment, fungicides\n"
        "• **Sustainable farming** — eco-friendly practices and IPM\n"
        "• **Crop health** — nutrition, irrigation, and plant care\n"
        "• **Specific detected diseases** — once a leaf is analysed, I can provide\n"
        "  tailored advice for the detected condition\n\n"
        "What would you like to know today?"
    ),
}


# ---------------------------------------------------------------------------
# Helper: append safety disclaimer
# ---------------------------------------------------------------------------
def _with_disclaimer(text: str) -> str:
    """Append the standard safety disclaimer to any advisory text."""
    return f"{text.strip()}\n\n{SAFETY_DISCLAIMER}"


# ---------------------------------------------------------------------------
# PUBLIC API: generate_advisory
# ---------------------------------------------------------------------------
def generate_advisory(
    disease: str,
    cause: str = "",
    symptoms: str = "",
    medication: str = "",
) -> dict:
    """
    Generate a structured AI advisory for a detected disease.

    Parameters
    ----------
    disease : str
        Internal disease class name (e.g. "Tomato_Late_blight").
    cause : str, optional
        Disease cause text from the knowledge base (for context).
    symptoms : str, optional
        Disease symptoms text from the knowledge base (for context).
    medication : str, optional
        Suggested medication from the knowledge base (for context).

    Returns
    -------
    dict
        Structured advisory with keys:
            farmer_explanation, prevention, sustainability,
            treatment_strategy, long_term_advice

    Notes
    -----
    # IBM BOB INTEGRATION POINT
    # To use IBM BOB for advisory generation, replace the local knowledge-base
    # lookup below with an API call, for example:
    #
    #   payload = {
    #       "disease": disease,
    #       "cause": cause,
    #       "symptoms": symptoms,
    #       "medication": medication,
    #   }
    #   response = requests.post(
    #       IBM_BOB_API_URL,
    #       headers={"Authorization": f"Bearer {IBM_BOB_API_KEY}"},
    #       json=payload,
    #       timeout=15,
    #   )
    #   if response.ok:
    #       return response.json()   # IBM BOB returns a matching dict structure
    #   # Fall back to local KB if API fails
    """
    entry = _ADVISORY_KB.get(disease, _FALLBACK_ADVISORY)

    return {
        "farmer_explanation": _with_disclaimer(entry["farmer_explanation"]),
        "prevention": _with_disclaimer(entry["prevention"]),
        "sustainability": _with_disclaimer(entry["sustainability"]),
        "treatment_strategy": _with_disclaimer(entry["treatment_strategy"]),
        "long_term_advice": _with_disclaimer(entry["long_term_advice"]),
    }


# ---------------------------------------------------------------------------
# PUBLIC API: chat_with_advisor
# ---------------------------------------------------------------------------
def chat_with_advisor(
    user_question: str,
    detected_disease: str = "",
    chat_history: list = None,
) -> str:
    """
    Generate a chatbot response to the user's question.

    Parameters
    ----------
    user_question : str
        The farmer's question in plain text.
    detected_disease : str, optional
        The internal class name of the last detected disease.
        When provided, the advisor uses it as smart context so that
        vague references like "this disease" or "prevent this" are understood.
    chat_history : list, optional
        List of previous {"role": ..., "content": ...} dicts (unused by
        local KB but reserved for IBM BOB stateful conversation context).

    Returns
    -------
    str
        Advisor response string with disclaimer appended.

    Notes
    -----
    # IBM BOB INTEGRATION POINT
    # To use IBM BOB for chatbot responses, replace the local logic below with:
    #
    #   payload = {
    #       "question": user_question,
    #       "context": {
    #           "detected_disease": detected_disease,
    #           "chat_history": chat_history or [],
    #       },
    #   }
    #   response = requests.post(
    #       IBM_BOB_API_URL + "/chat",
    #       headers={"Authorization": f"Bearer {IBM_BOB_API_KEY}"},
    #       json=payload,
    #       timeout=15,
    #   )
    #   if response.ok:
    #       return _with_disclaimer(response.json().get("answer", ""))
    #   # Fall back to local KB on API failure
    """
    question_lower = user_question.lower().strip()

    # ------------------------------------------------------------------
    # Greeting detection
    # ------------------------------------------------------------------
    greeting_words = {"hello", "hi", "hey", "namaste", "good morning", "good evening"}
    if any(word in question_lower for word in greeting_words):
        return _with_disclaimer(_GENERAL_KB["hello"])

    # ------------------------------------------------------------------
    # Smart context: resolve "this", "it", "the disease" to detected disease
    # ------------------------------------------------------------------
    context_phrases = [
        "this disease",
        "this problem",
        "this infection",
        "this",
        "it",
        "the disease",
        "the infection",
    ]
    uses_context_reference = any(ph in question_lower for ph in context_phrases)

    # Determine which disease to use for disease-specific Q&A
    effective_disease = detected_disease if detected_disease else ""

    # ------------------------------------------------------------------
    # Disease-specific keyword matching
    # ------------------------------------------------------------------
    # Determine intent: prevent, treat, fungicide/bactericide
    intent = None
    if any(kw in question_lower for kw in ["prevent", "avoid", "stop", "protect"]):
        intent = "prevent"
    elif any(kw in question_lower for kw in ["treat", "cure", "fix", "remove", "control"]):
        intent = "treat"
    elif any(kw in question_lower for kw in ["fungicide", "bactericide", "chemical", "spray", "medicine", "pesticide"]):
        intent = "fungicide"

    # Try to extract disease name from question if no context disease
    if not effective_disease:
        disease_name_map = {
            "late blight": "Tomato_Late_blight",
            "early blight": "Tomato_Early_blight",
            "bacterial spot": "Tomato_Bacterial_spot",
            "leaf mold": "Tomato_Leaf_Mold",
            "leaf mould": "Tomato_Leaf_Mold",
            "septoria": "Tomato_Septoria_leaf_spot",
            "spider mite": "Tomato_Spider_mites_Two_spotted_spider_mite",
            "mite": "Tomato_Spider_mites_Two_spotted_spider_mite",
            "target spot": "Tomato_Target_Spot",
            "yellow leaf curl": "Tomato_Tomato_YellowLeaf_Curl_Virus",
            "mosaic virus": "Tomato_Tomato_mosaic_virus",
            "mosaic": "Tomato_Tomato_mosaic_virus",
            "tomato mosaic": "Tomato_Tomato_mosaic_virus",
        }
        for keyword, class_name in disease_name_map.items():
            if keyword in question_lower:
                effective_disease = class_name
                break

    # Look up disease-specific intent response
    if effective_disease and intent:
        disease_responses = _CHATBOT_KB.get(effective_disease, {})
        if intent in disease_responses:
            prefix = ""
            if uses_context_reference and detected_disease:
                display = detected_disease.replace("Tomato_", "Tomato ").replace("_", " ")
                prefix = f"*Based on the detected disease: **{display}***\n\n"
            return _with_disclaimer(prefix + disease_responses[intent])

    # ------------------------------------------------------------------
    # General topic matching
    # ------------------------------------------------------------------
    topic_map = {
        "sustainable": "sustainable",
        "sustainability": "sustainable",
        "eco": "sustainable",
        "organic": "organic",
        "crop health": "crop health",
        "plant health": "crop health",
        "improve": "crop health",
        "ipm": "ipm",
        "integrated pest": "ipm",
        "fungicide": "fungicide",
        "fungicides": "fungicide",
        "chemical": "fungicide",
        "spray": "fungicide",
        "irrigation": "irrigation",
        "water": "irrigation",
        "watering": "irrigation",
        "rotation": "rotation",
        "rotate": "rotation",
        "crop rotation": "rotation",
    }
    for keyword, topic in topic_map.items():
        if keyword in question_lower:
            return _with_disclaimer(_GENERAL_KB[topic])

    # ------------------------------------------------------------------
    # Disease advisory lookup without specific intent
    # ------------------------------------------------------------------
    if effective_disease:
        advisory = generate_advisory(effective_disease)
        prefix = ""
        if uses_context_reference and detected_disease:
            display = detected_disease.replace("Tomato_", "Tomato ").replace("_", " ")
            prefix = f"*Based on the detected disease: **{display}***\n\n"
        combined = (
            f"{prefix}Here is a comprehensive overview for this condition:\n\n"
            f"**Farmer Explanation:**\n{advisory['farmer_explanation']}\n\n"
            f"**Prevention:**\n{advisory['prevention']}\n\n"
            f"**Treatment Strategy:**\n{advisory['treatment_strategy']}"
        )
        # Disclaimer is already embedded by generate_advisory; strip and re-add once
        combined = combined.replace(f"\n\n{SAFETY_DISCLAIMER}", "")
        return _with_disclaimer(combined)

    # ------------------------------------------------------------------
    # Fallback: generic helpful response
    # ------------------------------------------------------------------
    return _with_disclaimer(
        "I'm here to help with tomato disease management and sustainable farming. "
        "You can ask me about:\n\n"
        "• Preventing or treating specific diseases (e.g. 'How do I prevent Late Blight?')\n"
        "• Fungicide or bactericide recommendations\n"
        "• Sustainable and organic farming practices\n"
        "• Crop rotation, irrigation, and general crop health\n"
        "• IPM (Integrated Pest Management)\n\n"
        "For best results, upload a leaf image first — then I can give you advice "
        "tailored to the detected disease."
    )
