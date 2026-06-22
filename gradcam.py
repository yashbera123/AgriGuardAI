import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model):

    base_model = model.get_layer("mobilenetv2_1.00_224")

    last_conv_layer_name = "Conv_1"

    grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        base_model.get_layer(last_conv_layer_name).output,
        model.layers[-1].output
    ]
)
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(
        heatmap,
        0
    ) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap):

    heatmap = cv2.resize(
        heatmap,
        (
            original_image.shape[1],
            original_image.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        original_image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return superimposed_img