import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)
from tensorflow.keras.preprocessing import image_dataset_from_directory

# -----------------------------
# Configuration
# -----------------------------

DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# -----------------------------
# Load Dataset
# -----------------------------

print("Loading dataset...")

train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nClasses Found:")
for cls in class_names:
    print(cls)

# -----------------------------
# MobileNetV2 Preprocessing
# -----------------------------

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)

# -----------------------------
# Load Pretrained MobileNetV2
# -----------------------------

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -----------------------------
# Build Model
# -----------------------------

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# -----------------------------
# Compile Model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------

print("\nStarting Training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# Evaluate Model
# -----------------------------

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(val_ds)

print(f"\nValidation Accuracy: {accuracy:.4f}")

# -----------------------------
# Save Model
# -----------------------------

model.save(
    "model/tomato_disease_model.keras"
)

print("\nModel Saved Successfully!")
print("Saved to: model/tomato_disease_model.keras")