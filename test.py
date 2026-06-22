import tensorflow as tf

model = tf.keras.models.load_model(
    "model/tomato_disease_model.keras"
)

for layer in model.layers:
    print(layer.name)