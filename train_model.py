
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path(
    r"C:\Users\hp\Downloads\aptos2019-blindness-detection"
)

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_CSV = PROJECT_DIR / "dataset" / "train_split.csv"
VAL_CSV = PROJECT_DIR / "dataset" / "val_split.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 5

MODEL_DIR = PROJECT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load training and validation CSV files
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))

# --------------------------------------------------
# Calculate class weights
# --------------------------------------------------

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3, 4]),
    y=train_df["diagnosis"].to_numpy()
)

class_weight_dict = {
    i: float(weight)
    for i, weight in enumerate(class_weights)
}

print("\nClass weights:")
for class_id, weight in class_weight_dict.items():
    print(f"Class {class_id}: {weight:.4f}")

# --------------------------------------------------
# Load and preprocess images
# --------------------------------------------------

def load_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image_bytes,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # EfficientNetB0 includes its own rescaling layers.
    # Keep pixel values in the 0-255 range.
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.int32)

    return image, label


# --------------------------------------------------
# Create TensorFlow datasets
# --------------------------------------------------

def make_dataset(dataframe, training=False):
    paths = dataframe["image_path"].astype(str).values

    labels = dataframe["diagnosis"].astype(
        "int32"
    ).values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


train_ds = make_dataset(
    train_df,
    training=True
)

val_ds = make_dataset(
    val_df,
    training=False
)

# --------------------------------------------------
# Build EfficientNetB0 with pretrained ImageNet weights
# --------------------------------------------------

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Initially freeze the pretrained feature extractor.
base_model.trainable = False

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

features = base_model(
    inputs,
    training=False
)

features = tf.keras.layers.GlobalAveragePooling2D()(
    features
)

features = tf.keras.layers.Dropout(0.3)(
    features
)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(features)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)

# --------------------------------------------------
# Compile model
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# Training callbacks
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(
            MODEL_DIR / "best_retinal_model.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
]

# --------------------------------------------------
# Train using class weights
# --------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# --------------------------------------------------
# Save final model
# --------------------------------------------------

model.save(
    MODEL_DIR / "retinal_model.keras"
)

print("\nTraining finished.")
print(
    "Best checkpoint:",
    MODEL_DIR / "best_retinal_model.keras"
)
print(
    "Final model:",
    MODEL_DIR / "retinal_model.keras"
)
