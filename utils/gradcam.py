"""
gradcam.py — GradCAM Explainable AI for AgriGuard AI
======================================================

Public API:
    make_gradcam_heatmap(img_array, model)       -> numpy heatmap
    overlay_heatmap(original_image, heatmap)     -> BGR overlay image
    generate_gradcam_safe(image, model, config)  -> (overlay_pil | None, heatmap | None)
    generate_gradcam_explanation(heatmap, class_name, confidence) -> str
"""

import logging

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Original functions (UNCHANGED)
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model):
    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer_name = "Conv_1"

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            model.layers[-1].output,
        ],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    return superimposed_img


# ---------------------------------------------------------------------------
# NEW: Safe wrapper — returns (PIL overlay, raw heatmap) or (None, None)
# ---------------------------------------------------------------------------
def generate_gradcam_safe(
    pil_image: Image.Image,
    model,
    crop_config: dict,
) -> tuple:
    """
    Generate GradCAM overlay safely without crashing the app.

    Returns
    -------
    (overlay_pil, heatmap) on success
    (None, None) on any error
    """
    try:
        image_size = crop_config.get("image_size", (224, 224))
        resized = pil_image.convert("RGB").resize(image_size)
        img_array = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)

        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        img_array = preprocess_input(img_array.copy())

        # ── Dry forward pass to materialise Sequential outputs ──────────────
        # Sequential models are lazy — their intermediate layer outputs are
        # not available until the model has been called at least once.
        img_tensor = tf.constant(img_array)
        _ = model(img_tensor, training=False)  # side-effect: builds output graph

        # ── Build GradCAM sub-model ─────────────────────────────────────────
        base_model = model.get_layer("mobilenetv2_1.00_224")
        last_conv_layer = base_model.get_layer("Conv_1")

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_outputs[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        max_val = tf.math.reduce_max(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (max_val + 1e-8)
        heatmap = heatmap.numpy()

        # ── Overlay heatmap on original image ───────────────────────────────
        original_np = np.array(pil_image.convert("RGB").resize(image_size))
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        overlay_bgr = overlay_heatmap(original_bgr, heatmap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb)

        return overlay_pil, heatmap

    except Exception as exc:  # noqa: BLE001
        logger.warning("GradCAM generation failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# NEW: Natural-language explanation from heatmap analysis
# ---------------------------------------------------------------------------
# Disease-specific region vocabulary
_REGION_VOCAB = {
    "Tomato_Late_blight": "water-soaked lesion regions",
    "Tomato_Early_blight": "concentric ring lesion areas",
    "Tomato_Bacterial_spot": "necrotic spot clusters",
    "Tomato_Leaf_Mold": "mold-affected leaf surfaces",
    "Tomato_Septoria_leaf_spot": "small circular spot patterns",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "stippling and webbing zones",
    "Tomato_Target_Spot": "bullseye lesion areas",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": "yellowing and curl margins",
    "Tomato_Tomato_mosaic_virus": "mosaic discolouration zones",
    "Tomato_healthy": "uniform healthy tissue",
}

_QUADRANT_NAMES = {
    (0, 0): "upper-left",
    (0, 1): "upper-right",
    (1, 0): "lower-left",
    (1, 1): "lower-right",
}


def generate_gradcam_explanation(
    heatmap: np.ndarray,
    class_name: str,
    confidence: float,
) -> str:
    """
    Analyse heatmap activation distribution and return a farmer-friendly
    explanation of which regions the CNN focused on.

    Parameters
    ----------
    heatmap     : 2D numpy array of activation values (0–1)
    class_name  : internal disease class name
    confidence  : model confidence percentage (0–100)

    Returns
    -------
    Human-readable string describing CNN focus regions.
    """
    if heatmap is None or heatmap.size == 0:
        return (
            f"The AI model analysed the leaf with {confidence:.1f}% confidence "
            "and identified disease markers across the image."
        )

    h, w = heatmap.shape
    mid_h, mid_w = h // 2, w // 2

    # Compute mean activation per quadrant
    quadrant_means = {
        (0, 0): float(heatmap[:mid_h, :mid_w].mean()),
        (0, 1): float(heatmap[:mid_h, mid_w:].mean()),
        (1, 0): float(heatmap[mid_h:, :mid_w].mean()),
        (1, 1): float(heatmap[mid_h:, mid_w:].mean()),
    }

    # Top 2 activated quadrants
    sorted_quads = sorted(quadrant_means.items(), key=lambda x: x[1], reverse=True)
    top_quad = _QUADRANT_NAMES[sorted_quads[0][0]]
    second_quad = _QUADRANT_NAMES[sorted_quads[1][0]]
    top_pct = round(sorted_quads[0][1] * 100)
    second_pct = round(sorted_quads[1][1] * 100)

    region_term = _REGION_VOCAB.get(class_name, "affected tissue regions")

    # Overall focus concentration
    flat = heatmap.flatten()
    high_activation_pct = round(float((flat > 0.6).sum() / flat.size) * 100)

    if high_activation_pct > 40:
        spread = "widespread activation across the leaf surface"
    elif high_activation_pct > 20:
        spread = "moderately concentrated focus regions"
    else:
        spread = "tightly focused activation on specific lesions"

    explanation = (
        f"The CNN model focused primarily on the **{region_term}** "
        f"in the **{top_quad}** region ({top_pct}% activation intensity) "
        f"and the **{second_quad}** region ({second_pct}% activation intensity). "
        f"The heatmap shows {spread}, consistent with "
        f"a **{confidence:.1f}%** confidence prediction. "
        f"Brighter red areas in the heatmap indicate the pixels that most "
        f"influenced the AI's decision."
    )

    return explanation
